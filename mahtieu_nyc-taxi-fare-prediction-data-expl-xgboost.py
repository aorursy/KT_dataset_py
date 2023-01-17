%env JOBLIB_TEMP_FOLDER=/tmp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb # XGBoost package
#from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt # Matplotlib
%matplotlib inline

from datetime import datetime
from dateutil import tz

from geopy import distance
# Let's load 700_000 rows and exclude the first column
train = pd.read_csv("../input/train.csv", parse_dates=['pickup_datetime'], usecols=range(1,8), nrows=700_000)
test = pd.read_csv("../input/test.csv", parse_dates=['pickup_datetime'])
print("Train shape: {}".format(train.shape))
train.describe()
#Drop rows with null values
train = train.dropna(how = 'any', axis = 'rows')

#Free rides, negative fares and passenger count filtering
train = train[train.eval('(fare_amount > 0) & (passenger_count <= 6)')]
train.iloc[:100000].plot.scatter('pickup_longitude', 'pickup_latitude')
train.iloc[:100000].plot.scatter('dropoff_longitude', 'dropoff_latitude')
# Coordinates filtering
train = train[(train.pickup_longitude >= -77) &
              (train.pickup_longitude <= -70) &
              (train.dropoff_longitude >= -77) &
              (train.dropoff_longitude <= 70) &
              (train.pickup_latitude >= 35) &
              (train.pickup_latitude <= 45) &
              (train.dropoff_latitude >= 35) &
              (train.dropoff_latitude <= 45)
             ]
train.pickup_datetime = train.pickup_datetime.dt.tz_localize('UTC')
train.pickup_datetime = train.pickup_datetime.dt.tz_convert(tz.gettz('America/New_York'))

# Fares may change every year
train['year'] = train.pickup_datetime.dt.year

# Different fares during weekdays and weekends
train['dayofweek'] = train.pickup_datetime.dt.dayofweek

# Different fares during public holidays
train['dayofyear'] = train.pickup_datetime.dt.dayofyear

# Different fares in peak periods and off-peak periods
train['hourofday'] = train.pickup_datetime.dt.hour

train = train.drop('pickup_datetime', axis=1)
# Computes the distance (in miles) between the pickup and the dropoff locations
train['distance'] = train.apply(
    lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (x.dropoff_latitude, x.dropoff_longitude)).miles,
    axis = 1)

train = train[train.eval('(distance > 0) & (distance < 150)')]
fare_distance_ratio = (train.fare_amount/train.distance)
fare_distance_ratio.describe()

(fare_distance_ratio[fare_distance_ratio < 45]).hist()

# Drop incoherent fares
train = train[fare_distance_ratio < 45]
del fare_distance_ratio
# Coordinates of the 3 airpots of NYC
airports = {'jfk': [40.6441666, -73.7822222],
            'laguardia': [40.7747222, -73.8719444],
            'newark': [40.6897222, -74.175]}

# Computes the distance between the pickup location and the airport
pickup = train.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('jfk'))).miles, axis=1)
# Computes the distance between the dropoff location and the airport
dropoff = train.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('jfk'))).miles, axis=1)
# Selects the shortest distance
train['to_jfk'] = pd.concat((pickup, dropoff), axis=1).min(axis=1)

pickup = train.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('laguardia'))).miles, axis=1)
dropoff = train.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('laguardia'))).miles, axis=1)
train['to_laguardia'] = pd.concat((pickup, dropoff), axis=1).min(axis=1)

pickup = train.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('newark'))).miles, axis=1)
dropoff = train.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('newark'))).miles, axis=1)
train['to_newark'] = pd.concat((pickup, dropoff), axis=1).min(axis=1)

del pickup, dropoff
y = train.fare_amount
train = train.drop('fare_amount', axis=1)
#train = train.drop('passenger_count', axis=1)
# Grid of parameters for XGBoost training
#param_grid = {'n_estimators': [3000],
#              'max_depth': [7, 8, 9],
#              'learning_rate': [0.01, 0.1],
#              'subsample': [0.8, 0.9, 1],
#              'colsample_bytree': [0.8, 0.9, 1],
#              'gamma': [0, 1e-5, 1e-4, 1e-3],
#              'reg_alpha': [1e-4]
#              }


#xgb_grid_search = GridSearchCV(xgb.XGBRegressor(eval_metric='rmse'),
#                               param_grid=param_grid,
#                               cv=3,
#                               n_jobs=-1,
#                               verbose=0)

#xgb_grid_search.fit(train, y)
#print("Best estimator: {}".format(xgb_grid_search.best_params_))
#print("Best score: {}".format(xgb_grid_search.best_score_))

# The best parameters given by the grid search
xgb_param = {'eval_metric': 'rmse',
            'n_estimators': 3000,
            'max_depth': 9,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.8,
            'gamma': 1e-4,
            'reg_alpha': 1e-4,
            'verbose': 0,
            'n_jobs': -1
            }

xgb_model = xgb.XGBRegressor(**xgb_param)
xgb_model.fit(train, y)
xgb.plot_importance(xgb_model)
# Processing
test_key = test['key']
test = test.drop('key', axis=1)
#test = test.drop('passenger_count', axis=1)
test.pickup_datetime = test.pickup_datetime.dt.tz_localize('UTC')
test.pickup_datetime = test.pickup_datetime.dt.tz_convert(tz.gettz('America/New_York'))

test['year'] = test.pickup_datetime.dt.year
test['dayofweek'] = test.pickup_datetime.dt.dayofweek
test['dayofyear'] = test.pickup_datetime.dt.dayofyear
test['hourofday'] = test.pickup_datetime.dt.hour
test = test.drop('pickup_datetime', axis=1)


test['distance'] = test.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (x.dropoff_latitude, x.dropoff_longitude)).miles, axis = 1)

pickup = test.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('jfk'))).miles, axis=1)
dropoff = test.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('jfk'))).miles, axis=1)
test['to_jfk'] = pd.concat((pickup, dropoff), axis = 1).min(axis=1)
pickup = test.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('laguardia'))).miles, axis=1)
dropoff = test.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('laguardia'))).miles, axis=1)
test['to_laguardia'] = pd.concat((pickup, dropoff), axis = 1).min(axis=1)
pickup = test.apply(lambda x: distance.distance((x.pickup_latitude, x.pickup_longitude), (airports.get('newark'))).miles, axis=1)
dropoff = test.apply(lambda x: distance.distance((x.dropoff_latitude, x.dropoff_longitude), (airports.get('newark'))).miles, axis=1)
test['to_newark'] = pd.concat((pickup, dropoff), axis = 1).min(axis=1)
del pickup, dropoff
xgb_predict = xgb_model.predict(test)
xgb_submission = pd.DataFrame({ 'key': test_key,
                               'fare_amount': xgb_predict })
xgb_submission.to_csv("xgb_submission.csv", index=False)