# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# References : https://www.kaggle.com/manojvijayan/feature-engineering-and-xgboost

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('fivethirtyeight')

import geopy.distance

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc


#=====================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Any results you write to the current directory are saved as output.
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=20_00_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=20_00_000, low_memory=True)
    return train,test
train, test = load_Data()
train.head(5)
train.describe()
train.info()
train.isnull().sum()
train = train.fillna(0)
train['key2'] = pd.to_datetime(train['key'], errors='coerce')
train['key2'].head()
train.info()
test['key2'] = pd.to_datetime(test['key'], errors='coerce')
train['fare_amount'].plot(kind='box')
gc.collect()
train.describe()
print("% of fares above 25$ - {:0.2f}".format(train[train['fare_amount'] > 25]['key'].count()*100/train['key'].count()))
print("% of fares above 50$ - {:0.2f}".format(train[train['fare_amount'] > 50]['key'].count()*100/train['key'].count()))
print("% of fares above 100$ - {:0.2f}".format(train[train['fare_amount'] > 100]['key'].count()*100/train['key'].count()))
print("% of fares below 0$ - {:0.2f}".format(train[train['fare_amount'] < 0]['key'].count()*100/train['key'].count()))
# fig, axxx = plt.subplot(2,2, figsize=(20,10))
# train[~(train['fare_amount'] > 25)]['fare_amount'].plot(kind='box', ax = axxx[0][0])
# train[~(train['fare_amount'] > 50)]['fare_amount'].plot(kind='box', ax = axxx[0][1])
# train[~(train['fare_amount'] > 100)]['fare_amount'].plot(kind='box', ax = axxx[1][0])
# train[~(train['fare_amount'] < 0 )]['fare_amount'].plot(kind='box', ax = axxx[1][1])

fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train['fare_amount'] > 25)]['fare_amount'].plot(kind="box",ax=axarr[0][0])
train[~(train['fare_amount'] > 50)]['fare_amount'].plot(kind="box",ax=axarr[0][1])
train[~(train['fare_amount'] > 100)]['fare_amount'].plot(kind="box",ax=axarr[1][0])
train[~(train['fare_amount'] < 0 )]['fare_amount'].plot(kind='box',ax = axarr[1][1])
train['passenger_count'].plot(kind='box')
print("Count of invalid pickup latitude", train[(train['pickup_latitude'] > 90) | (train['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", train[(train['dropoff_latitude'] > 90) | (train['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", train[(train['pickup_longitude'] > 180) | (train['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", train[(train['dropoff_longitude'] > 180) | (train['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())
print("Count of invalid pickup latitude", test[(test['pickup_latitude'] > 90) | (test['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", test[(test['dropoff_latitude'] > 90) | (test['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", test[(test['pickup_longitude'] > 180) | (test['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", test[(test['dropoff_longitude'] > 180) | (test['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())
train = train[~((train['pickup_latitude'] > 90) | (train['pickup_latitude'] < -90))]
train = train[~((train['dropoff_latitude'] > 90) | (train['dropoff_latitude'] < -90))]
train = train[~((train['pickup_longitude'] > 180) | (train['pickup_longitude'] < -180))]
train = train[~((train['dropoff_longitude'] > 180) | (train['dropoff_longitude'] < -180))]
train['distance']=train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)
train.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)
test['distance']=test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)
test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)
print("% of trips above 25 KM - {:0.2f}".format(train[train['distance'] > 25]['key'].count()*100/train.count()['key']))
train['year'] = train['key2'].apply(lambda x : x.year)
train['month'] = train['key2'].apply(lambda x : x.month)
train['day'] = train['key2'].apply(lambda x : x.day)
train['day of week'] = train['key2'].apply(lambda x : x.weekday())
train['hour'] = train['key2'].apply(lambda x : x.hour)
train["week"] = train['key2'].apply(lambda x: x.week)
train['day_of_year'] = train['key2'].apply(lambda x:x.dayofyear)
train['week_of_year'] = train['key2'].apply(lambda x:x.weekofyear)
train['quarter'] = train['key2'].apply(lambda x:x.quarter)
train.columns
test['year'] = test['key2'].apply(lambda x : x.year)
test['month'] = test['key2'].apply(lambda x : x.month)
test['day'] = test['key2'].apply(lambda x : x.day)
test['day of week'] = test['key2'].apply(lambda x : x.weekday())
test['hour'] = test['key2'].apply(lambda x : x.hour)
test["week"] = test['key2'].apply(lambda x: x.week)
test['day_of_year'] = test['key2'].apply(lambda x:x.dayofyear)
test['week_of_year'] = test['key2'].apply(lambda x:x.weekofyear)
test['quarter'] = test['key2'].apply(lambda x:x.quarter)
column_list = ['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour','week','day_of_year', 'week_of_year', 'quarter']
y_train_ = train['fare_amount']
X_train_ = train.drop(['fare_amount'], axis=1)
X_train_ = train[column_list] 
X_test = test[column_list]
X_train_.shape,y_train_.shape,X_test.shape
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.1)

X_train.shape,X_val.shape,y_train.shape,y_val.shape
import xgboost as xgb
from bayes_opt import BayesianOptimization
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_val)
# def xgb_evaluate(max_depth, gamma, min_child_weight, max_delta_step, subsample,colsample_bytree, eta):
#     params = {'eval_metric': 'rmse',
#               'max_depth': int(max_depth),
#               'subsample': 0.8,
#               'eta': eta,
#               'gamma': gamma,
#               'colsample_bytree': colsample_bytree,
#               'min_child_weight': min_child_weight,
#               'max_delta_step':max_delta_step
#              }
#     # Used around 1000 boosting rounds in the full model
#     cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
#     # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
#     return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
# xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (1, 15), 
#                                              'max_depth': (2, 12),
#                                              'gamma': (0.001, 10.0),
#                                              'min_child_weight': (0, 20),
#                                              'max_delta_step': (0, 10),
#                                              'subsample': (0.4, 1.0),
#                                              'colsample_bytree' :(0.4, 1.0),
#                                              'eta': (0.01,0.1)
#                                             })
# # Use the expected improvement acquisition function to handle negative numbers
# # Optimally needs quite a few more initiation points and number of iterations
# xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
# params = xgb_bo.res['max']['max_params']
# params['max_depth'] = int(params['max_depth'])
params = {'colsample_bytree': 1.0,
 'eta': 0.1,
 'gamma': 0.001,
 'max_delta_step': 10.0,
 'max_depth': 12,
 'min_child_weight': 20.0,
 'subsample': 1.0}
from sklearn.metrics import mean_squared_error
# Train a new model with the best parameters from the search
model2 = xgb.train(params, dtrain, num_boost_round=250)

# Predict on testing and training set
y_pred = model2.predict(dtest)
y_train_pred = model2.predict(dtrain)

# Report testing and training RMSE
print(np.sqrt(mean_squared_error(y_val, y_pred)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
fscores = pd.DataFrame({'X': list(model2.get_fscore().keys()), 'Y': list(model2.get_fscore().values())})
fscores.sort_values(by='Y').plot.bar(x='X')
dtest = xgb.DMatrix(X_test)
y_pred_test = model2.predict(dtest)
sub = pd.read_csv("../input/sample_submission.csv")
sub["Target"] = y_pred_test
sub.to_csv('submission.csv', index=False)