from typing import Optional, Union



import holoviews as hv

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.utils.validation import check_array

from sklearn.base import TransformerMixin, BaseEstimator, ClusterMixin

from sklearn.mixture._base import BaseMixture

from sklearn.utils import check_random_state

from sklearn.utils.validation import check_is_fitted



from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import FunctionTransformer

from sklearn.base import clone

from sklearn.ensemble._base import BaseEnsemble, _set_random_states

from sklearn.linear_model._base import LinearModel



from statsmodels.gam.smooth_basis import _eval_bspline_basis

from statsmodels.gam.tests.test_penalized import df_autos





hv.extension('bokeh')



class FeatureSampler(BaseEstimator, TransformerMixin):

    def __init__(self, n_features: int = 1, random_state=None):

        self.n_features = n_features

        self.random_state = random_state



    def fit(self, X: np.ndarray, y=None):

        self.n_features_ = min(X.shape[1], self.n_features)

        self.random_state_ = check_random_state(self.random_state)

        self.feature_ = np.random.choice(X.shape[1], self.n_features)



        return self



    def transform(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(self)

        check_array(X)



        if isinstance(X, np.ndarray):

            return X[:, self.feature_]

        elif isinstance(X, pd.DataFrame):

            return X.iloc[:, self.feature_]





class TruncatedPowerBasis(BaseEstimator, TransformerMixin):

    def __init__(

        self,

        degree: int = 3,

        knots: Optional[Union[int, np.ndarray, ClusterMixin, BaseMixture]] = None,

        random_state=None,

    ):

        self.knots = knots

        self.degree = degree

        self.random_state = random_state



    def fit(self, X: np.ndarray, y=None):

        if not isinstance(self.degree, int) or self.degree < 0:

            raise ValueError("Must be an integer greater or equal to 0")



        self.random_state_ = check_random_state(self.random_state)

        check_array(X)



        if X.shape[1] > 1:

            raise ValueError(

                "TruncatedPowerBasis only accepts 2D arrays with 1 feature, this has shape %s"

                % X.shape[1]

            )



        if isinstance(self.knots, ClusterMixin) or isinstance(self.knots, BaseMixture):

            self.knots.fit(X)



            if hasattr(self.knots, "cluster_centers_"):

                self.knots_ = self.knots.cluster_centers_.reshape(1, -1)

            elif hasattr(self.knots, "means_"):

                self.knots_ = self.knots.means_.reshape(1, -1)

            elif hasattr(self.knots, "medoids_"):

                self.knots_ = self.knots.medoids_.reshape(1, -1)

            else:

                raise TypeError(

                    "Model does not have cluster_centers_, means_ or medoids_ attributes."

                )

        elif isinstance(self.knots, int):

            indexes = self.random_state_.choice(X.shape[0], self.knots)



            self.knots_ = X[indexes, :].reshape(1, -1)

        elif self.knots is None and isinstance(X, np.ndarray):

            self.knots_ = np.unique(X.reshape(1, -1))

        elif self.knots is None and isinstance(X, pd.DataFrame):

            self.knots_ = np.unique(X.to_numpy().reshape(1, -1))

            name = X.columns.astype(str)[0]

            self.names_ = [

                f"|{name} - ({xi})|^{self.degree}"

                for xi in (self.knots_.round(3).astype(str).flatten().tolist())

            ]

        else:

            self.knots_ = np.unique(X.T)



        return self



    def transform(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(self)

        check_array(X)



        if X.shape[1] > 1:

            raise ValueError(

                "TruncatedPowerBasis only accepts 2D arrays with 1 feature"

            )



        if isinstance(X, np.ndarray):

            return (X - self.knots_) ** self.degree

        elif isinstance(X, pd.DataFrame):

            return (

                pd.DataFrame(X.to_numpy() - np.array(self.knots_), columns=self.names_)

                .abs()

                .pow(self.degree)

            )





class CoxdeBoorBasis(TruncatedPowerBasis):

    def transform(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(self)

        check_array(X)



        if X.shape[1] > 1:

            raise ValueError(

                "TruncatedPowerBasis only accepts 2D arrays with 1 feature"

            )



        if isinstance(X, np.ndarray):

            x = X.flatten()

            return _eval_bspline_basis(x, x, self.degree, 0)

        elif isinstance(X, pd.DataFrame):

            x = X.iloc[:, 0]

            return _eval_bspline_basis(x, x, self.degree, 0)





class GeneralizedAdditiveModel(BaseEnsemble):

    def __init__(

        self,

        base_estimator=LinearRegression(fit_intercept=False),

        estimator_params=tuple(),

        basis=TruncatedPowerBasis(1),

        max_iter=10,

        tol=1e-6,

    ):

        self.base_estimator = base_estimator

        self.basis = basis

        self.max_iter = max_iter

        self.tol = tol

        self.estimator_params = estimator_params



    def _make_estimator(self, feature, append=True, random_state=None):

        estimator = Pipeline(

            [

                (

                    "selector",

                    ColumnTransformer(

                        [(str(feature), FunctionTransformer(), [feature])]

                    ),

                ),

                ("basis", clone(self.basis_)),

                ("estimator", clone(self.base_estimator_)),

            ]

        )

        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})



        if random_state is not None:

            _set_random_states(estimator, random_state)



        if append:

            self.estimators_.append(estimator)



        return estimator



    def fit(self, X: np.ndarray, y: np.ndarray):

        self.n_features_ = X.shape[1]

        if isinstance(self.base_estimator, LinearModel):

            self.base_estimator_ = self.base_estimator

        else:

            raise TypeError("This is not an instance of a linear model")

        self.basis_ = self.basis



        self.intercept_ = y.mean()

        errors = y - self.intercept_



        self.estimators_ = []

        for model, feature in enumerate(range(self.n_features_)):

            self._make_estimator(feature)

            self.estimators_[model].fit(X, errors)

            self.estimators_[feature].named_steps["estimator"].intercept_ -= (

                self.estimators_[feature].predict(X).mean()

            )

            errors = y - self._predict(X)



        for step in range(self.max_iter):

            for model, feature in enumerate(range(self.n_features_)):

                # backfitting step

                errors = y - self._predict(X, exclude=model)

                self.estimators_[feature].fit(X, errors)



                # mean centering of estimated function

                self.estimators_[feature].named_steps["estimator"].intercept_ -= (

                    self.estimators_[feature].predict(X).mean()

                )

                prev_errors = errors



            if np.all(np.abs(prev_errors - errors) < self.tol):

                break



        return self



    def _predict(self, X: np.ndarray, exclude=None):

        if exclude is None:

            return (

                sum((estimator.predict(X) for estimator in self.estimators_))

                + self.intercept_

            )

        else:

            return (

                sum(

                    (

                        estimator.predict(X)

                        for i, estimator in enumerate(self.estimators_)

                        if i != exclude

                    )

                )

                + self.intercept_

            )



    def predict(self, X: np.ndarray):

        return self._predict(X)

df_autos.head()
df_autos.city_mpg.plot.kde()
X, y = (pd.concat([df_autos.loc[:,['weight','hp']], 

                   pd.get_dummies(df_autos.fuel),

                   pd.get_dummies(df_autos.drive)], axis=1),

                   df_autos.city_mpg.apply(np.log))

sns.pairplot(X.loc[:,['weight','hp']].assign(city_mpg = y))
gam = GeneralizedAdditiveModel(max_iter=50, base_estimator=LinearRegression(fit_intercept=False), basis=CoxdeBoorBasis())

gam.fit(X, y)

(gam.predict(X) - y).plot.kde(title='Distribution of Errors')
smooth_basis = pd.DataFrame({f'{name}_spline': model.predict(X) for name, model in zip(X.columns, gam.estimators_)}).assign(city_mpg = y)

smooth_basis.head()
sns.pairplot(smooth_basis.loc[:,['weight_spline', 'hp_spline', 'city_mpg']])