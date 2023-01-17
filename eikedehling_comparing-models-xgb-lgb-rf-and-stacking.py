import numpy as np

import pandas as pd

import haversine



# Read training data (first 100k rows only for speed..)

train = pd.read_csv('../input/train.csv', parse_dates=['pickup_datetime'], nrows=100000)



# Log transform the Y values

train_y = np.log1p(train['trip_duration'])



# Add some features..

train['distance'] = train.apply(lambda r: haversine.haversine((r['pickup_latitude'], r['pickup_longitude']), (r['dropoff_latitude'], r['dropoff_longitude'])), axis=1)

train['month'] = train.pickup_datetime.dt.month

train['day'] = train.pickup_datetime.dt.day

train['dw'] = train.pickup_datetime.dt.dayofweek

train['h'] = train.pickup_datetime.dt.hour

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)

train = train.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration'], axis=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Ridge, Lasso

from lightgbm.sklearn import LGBMRegressor

from xgboost.sklearn import XGBRegressor

from catboost import CatBoostRegressor



lgbm_model = LGBMRegressor(

    n_estimators=150,

    subsample=0.85,

    subsample_freq=5,

    learning_rate=0.05

)



catboost_model = CatBoostRegressor(iterations=150)

xgb_model = XGBRegressor(objective='reg:linear', n_estimators=150, subsample=0.75)

rf_model = RandomForestRegressor(n_estimators=25, min_samples_leaf=25, min_samples_split=25)

tree_model = DecisionTreeRegressor(min_samples_leaf=25, min_samples_split=25)

knn_model = KNeighborsRegressor(n_neighbors=25, weights='distance')

ridge_model = Ridge(alpha=75.0)

lasso_model = Lasso(alpha=0.75)
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin





class AveragingRegressor(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, regressors):

        self.regressors = regressors

        self.predictions = None



    def fit(self, X, y):

        for regr in self.regressors:

            regr.fit(X, y)

        return self



    def predict(self, X):

        self.predictions = np.column_stack([regr.predict(X) for regr in self.regressors])

        return np.mean(self.predictions, axis=1)

    

    

averaged_model = AveragingRegressor([catboost_model, xgb_model, rf_model, lgbm_model])
from mlxtend.regressor import StackingCVRegressor



stacked_model = StackingCVRegressor(

    regressors=[catboost_model, xgb_model, rf_model, lgbm_model],

    meta_regressor=Ridge()

)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer



def rmse_fun(predicted, actual):

    return np.sqrt(np.mean(np.square(predicted - actual)))



rmse = make_scorer(rmse_fun, greater_is_better=False)



models = [

     ('CatBoost', catboost_model),

     ('XGBoost', xgb_model),

     ('LightGBM', lgbm_model),

     ('DecisionTree', tree_model),

     ('RandomForest', rf_model),

     ('Ridge', ridge_model),

     ('Lasso', lasso_model),

     ('KNN', knn_model),

     ('Averaged', averaged_model),

     ('Stacked', stacked_model),

]



scores = [

    -1.0 * cross_val_score(model, train.values, train_y.values, scoring=rmse).mean()

    for _,model in models

]



dataz = pd.DataFrame({ 'Model': [name for name, _ in models], 'Error (RMSE)': scores })

dataz.plot(x='Model', kind='bar')