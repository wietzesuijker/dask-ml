import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pytest
import sklearn.linear_model

import dask_ml.datasets
import dask_ml.ensemble


class TestBlockwiseVotingClassifier:
    def test_hard_voting_array(self):
        X, y = dask_ml.datasets.make_classification(chunks=25)
        clf = dask_ml.ensemble.BlockwiseVotingClassifier(
            sklearn.linear_model.LogisticRegression(solver="lbfgs"),
            classes=[0, 1],
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 4

        X2, y2 = dask_ml.datasets.make_classification(chunks=20)

        result = clf.predict(X2)
        assert isinstance(result, da.Array)
        assert result.dtype == np.dtype("int64")
        assert result.shape == (len(y),)
        assert result.numblocks == y2.numblocks
        result_ = result.compute()
        assert result_.dtype == result.dtype
        assert result_.shape == result.shape

        with pytest.raises(AttributeError, match="hard"):
            clf.predict_proba

        score = clf.score(X2, y2)
        assert isinstance(score, float)

        # ndarray
        X3, y3 = dask.compute(X2, y2)
        result2 = clf.predict(X3)
        assert isinstance(result2, np.ndarray)
        da.utils.assert_eq(result, result2)
        score2 = clf.score(X3, y3)
        assert score == score2

        _, y4 = dask_ml.datasets.make_classification(chunks=20)
        with pytest.raises(ValueError, match="4 != 5"):
            clf.fit(X, y4)

    def test_bad_chunking_raises(self):
        X = da.ones((10, 5), chunks=3)
        y = da.ones(10, chunks=3)
        clf = dask_ml.ensemble.BlockwiseVotingClassifier(
            sklearn.linear_model.LogisticRegression(solver="lbfgs"),
            classes=[0, 1],
        )

        with pytest.raises(TypeError):
            # this should *really* be a ValueError...
            clf.fit(X, y)

    def test_hard_voting_frame(self):
        X, y = dask_ml.datasets.make_classification(chunks=25)
        X = dd.from_dask_array(X)
        y = dd.from_dask_array(y)

        clf = dask_ml.ensemble.BlockwiseVotingClassifier(
            sklearn.linear_model.LogisticRegression(solver="lbfgs"),
            classes=[0, 1],
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 4

        X2, y2 = dask_ml.datasets.make_classification(chunks=20)
        X2 = dd.from_dask_array(X2)
        y2 = dd.from_dask_array(y2)

        result = clf.predict(X2)
        assert isinstance(result, da.Array)  # TODO(pandas-IO)
        assert result.dtype == np.dtype("int64")
        assert len(result.shape) == 1 and np.isnan(result.shape[0])
        assert result.numblocks == (y2.npartitions,)
        result_ = result.compute()
        assert result_.dtype == result.dtype
        assert result_.shape == (len(y2),)

        with pytest.raises(AttributeError, match="hard"):
            clf.predict_proba

        score = clf.score(X2, y2)
        assert isinstance(score, float)

        # ndarray
        X3, y3 = dask.compute(X2, y2)
        result2 = clf.predict(X3)
        assert isinstance(result2, np.ndarray)
        da.utils.assert_eq(result, result2)
        # TODO: accuracy score raising for pandas.
        # score2 = clf.score(X3, y3)
        # assert score == score2

    def test_soft_voting_array(self):
        X, y = dask_ml.datasets.make_classification(chunks=25)
        clf = dask_ml.ensemble.BlockwiseVotingClassifier(
            sklearn.linear_model.LogisticRegression(solver="lbfgs"),
            voting="soft",
            classes=[0, 1],
        )
        clf.fit(X, y)

        assert len(clf.estimators_) == 4

        result = clf.predict(X)
        assert isinstance(result, da.Array)
        assert result.dtype == np.dtype("int64")
        assert result.shape == (len(X),)
        result_ = result.compute()
        assert result_.dtype == result.dtype
        assert result_.shape == result.shape

        result = clf.predict_proba(X)
        assert result.dtype == np.dtype("float64")
        assert result.shape == (len(X), 2)  # 2 classes
        assert result.numblocks == (4, 1)

        score = clf.score(X, y)
        assert isinstance(score, float)

    @pytest.mark.xfail(
        True,
        reason="AssertionError da.utils.assert_eq(result, result2)",
    )
    def test_soft_voting_frame(self):
        X, y = dask_ml.datasets.make_classification(chunks=25)
        X = dd.from_dask_array(X)
        y = dd.from_dask_array(y)

        clf = dask_ml.ensemble.BlockwiseVotingClassifier(
            sklearn.linear_model.LogisticRegression(solver="lbfgs"),
            voting="soft",
            classes=[0, 1],
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 4

        X2, y2 = dask_ml.datasets.make_classification(chunks=20)
        X2 = dd.from_dask_array(X2)
        y2 = dd.from_dask_array(y2)

        result = clf.predict(X2)
        assert isinstance(result, da.Array)  # TODO(pandas-IO)
        assert result.dtype == np.dtype("int64")
        assert len(result.shape) == 1 and np.isnan(result.shape[0])
        assert result.numblocks == (y2.npartitions,)
        result_ = result.compute()
        assert result_.dtype == result.dtype
        assert result_.shape == (len(y2),)

        result = clf.predict_proba(X2)
        assert result.dtype == np.dtype("float64")
        assert len(result.shape) == 2
        assert np.isnan(result.shape[0])
        assert result.shape[1] == 2
        assert result.numblocks == (5, 1)

        score = clf.score(X, y)
        assert isinstance(score, float)

        # ndarray
        X3, y3 = dask.compute(X2, y2)
        result2 = clf.predict_proba(X3)
        assert isinstance(result2, np.ndarray)
        da.utils.assert_eq(result, result2)
        # TODO: accuracy score raising for pandas.
        # score2 = clf.score(X3, y3)
        # assert score == score2

    def test_no_classes_raises(self):
        X, y = dask_ml.datasets.make_classification(chunks=25)
        clf = dask_ml.ensemble.BlockwiseVotingClassifier(
            sklearn.linear_model.LogisticRegression(solver="lbfgs"),
        )
        with pytest.raises(ValueError, match="classes"):
            clf.fit(X, y)


class TestBlockwiseVotingRegressor:
    def test_no_unnecessary_computation_in_fit(self, monkeypatch):
        X, y = dask_ml.datasets.make_regression(n_features=20, chunks=25)
        compute_called = False
        original_compute = X.compute

        def spy_compute(*args, **kwargs):
            nonlocal compute_called
            compute_called = True
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(X, "compute", spy_compute)

        est = dask_ml.ensemble.BlockwiseVotingRegressor(
            sklearn.linear_model.LinearRegression(),
        )
        est.fit(X, y)
        # Ensure that X.compute() was never invoked during fitting.
        assert compute_called is False
        # Verify that _n_samples was set using lazy metadata.
        assert est._n_samples == X.shape[0]

    def test_fit_array(self):
        X, y = dask_ml.datasets.make_regression(n_features=20, chunks=25)
        est = dask_ml.ensemble.BlockwiseVotingRegressor(
            sklearn.linear_model.LinearRegression(),
        )
        est.fit(X, y)
        assert len(est.estimators_) == 4

        X2, y2 = dask_ml.datasets.make_regression(n_features=20, chunks=20)
        result = est.predict(X2)
        assert result.dtype == np.dtype("float64")
        assert result.shape == y2.shape
        assert result.numblocks == y2.numblocks

        score = est.score(X2, y2)
        assert isinstance(score, float)

        # ndarray
        X3, y3 = dask.compute(X2, y2)
        result2 = est.predict(X3)
        assert isinstance(result2, np.ndarray)
        da.utils.assert_eq(result, result2)
        # TODO: r2_score raising for ndarray
        # score2 = est.score(X3, y3)
        # assert score == score2

    def test_fit_frame(self):
        X, y = dask_ml.datasets.make_regression(n_features=20, chunks=25)
        X = dd.from_dask_array(X)
        y = dd.from_dask_array(y)

        est = dask_ml.ensemble.BlockwiseVotingRegressor(
            sklearn.linear_model.LinearRegression(),
        )
        est.fit(X, y)
        assert len(est.estimators_) == 4

        X2, y2 = dask_ml.datasets.make_regression(n_features=20, chunks=20)
        result = est.predict(X2)
        assert result.dtype == np.dtype("float64")
        assert result.shape == y2.shape
        assert result.numblocks == y2.numblocks

        score = est.score(X2, y2)
        assert isinstance(score, float)

        # ndarray
        X3, y3 = dask.compute(X2, y2)
        result2 = est.predict(X3)
        assert isinstance(result2, np.ndarray)
        da.utils.assert_eq(result, result2)
        # TODO: r2_score raising for ndarray
        # score2 = est.score(X3, y3)
        # assert score == score2

    def test_predict_with_different_chunks(self):
        X, y = dask_ml.datasets.make_regression(n_features=20, chunks=25)
        est = dask_ml.ensemble.BlockwiseVotingRegressor(
            sklearn.linear_model.LinearRegression(),
        )
        est.fit(X, y)

        X_test, y_test = dask_ml.datasets.make_regression(n_features=20, chunks=20)
        result = est.predict(X_test)
        assert result.dtype == np.dtype("float64")
        assert result.shape == y_test.shape
        # Prediction is rechunked to have one block per estimator.
        assert result.numblocks[0] == len(est.estimators_)

        score = est.score(X_test, y_test)
        assert isinstance(score, float)

        X_test_np, y_test_np = dask.compute(X_test, y_test)
        result_np = est.predict(X_test_np)
        da.utils.assert_eq(result, result_np)
