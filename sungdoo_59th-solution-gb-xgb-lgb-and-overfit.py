import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
print("train.csv. Shape: ",train.shape)

print("test.csv. Shape: ",test.shape)
train_data = train

test_data = test

sub_id = test['id']
for i in [train_data,test_data]:

    i['date'] = i['date'].apply(lambda e: e.split('T')[0])

    i['yr_renovated'] = i['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    i['renovated'] = i['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    i['yr_renovated'] = i['yr_renovated'].fillna(i['yr_built'])

    i['renovated'] = i['renovated'].fillna(0)

    i['yr_renovated'] = i['yr_renovated'].astype('int')



train.loc[train.renovated > 0,'renovated']= 1.0

test.loc[test.renovated > 0,'renovated']= 1.0
change_columns = ['bedrooms', 'sqft_living', 'sqft_lot','sqft_above',

       'sqft_basement','sqft_living15', 'sqft_lot15']



for i in change_columns:

    train_data[i] = np.log1p(train[i].values)

    test_data[i] = np.log1p(test[i].values)
for df in [train_data,test_data]:

    # 방의 전체 갯수 

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # 거실의 비율 

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    # 총 면적

    df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

    # 면적 대비 거실의 비율 

    df['sqft_ratio_1'] = df['sqft_living'] / df['sqft_total_size']

    df['sqft_ratio15'] = df['sqft_living15'] / df['sqft_lot15']
train_data = train_data.drop(['id'], axis=1)

test_data = test_data.drop(['id'], axis=1)
train_data['date'] = pd.to_datetime(train_data['date'].astype('str'))

test_data['date'] = pd.to_datetime(test_data['date'].astype('str'))

for i in [train_data, test_data]:

    i['year'] = i['date'].dt.year

    i['month'] = i['date'].dt.month

    

train_data = train_data.drop('date', axis=1)

test_data = test_data.drop('date', axis=1)
x = train_data.loc[:,:]

y = train_data.loc[:,'price']

x_test = test_data.loc[:,:]

x = x.drop(['price'], axis=1)

log_y = np.log1p(y)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x.values)

    rmse= np.sqrt(-cross_val_score(model, x.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4, gamma=0.0468, 

                             learning_rate=0.05, max_depth=6, 

                             min_child_weight=1.7817, n_estimators=15000,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.8, silent=1,

                             random_state =25, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leave=2,

                              learning_rate=0.05, n_estimators=15000,

                              max_bin = 80, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_gb = GradientBoostingRegressor(n_estimators=15000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =25)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))

def rmse_expm1(y, y_pred):

    return np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred)))
model_xgb.fit(x, log_y)

xgb_train_pred = model_xgb.predict(x)

xgb_pred = model_xgb.predict(x_test)

xgb_pred = np.expm1(xgb_pred)

print(rmse_expm1(log_y, xgb_train_pred))
model_lgb.fit(x, log_y)

lgb_train_pred = model_lgb.predict(x)

lgb_pred = model_lgb.predict(x_test)

lgb_pred = np.expm1(lgb_pred)

print(rmse_expm1(log_y, lgb_train_pred))
model_gb.fit(x, log_y)

gb_train_pred = model_gb.predict(x)

gb_pred = model_gb.predict(x_test)

gb_pred = np.expm1(gb_pred)

print(rmse_expm1(log_y, gb_train_pred))
print('RMSE score on train data:')

print(rmse_expm1(log_y,xgb_train_pred*0.4 + lgb_train_pred*0.4 + gb_train_pred*0.2 ))
ensemble = xgb_pred*0.4 + lgb_pred*0.4 + gb_pred*0.2
sub1 = pd.DataFrame(data={'id':sub_id,'price':ensemble})

sub1.to_csv('submission1.csv', index=False)