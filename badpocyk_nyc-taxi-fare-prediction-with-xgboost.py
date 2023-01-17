from pandas import read_csv, DataFrame, to_datetime
from numpy import radians, sin, cos, arcsin, sqrt, mean
from sklearn import ensemble
import time
from datetime import datetime
import xgboost as xgb


train_data = read_csv("../input/train.csv")
test = read_csv("../input/test.csv")
sample_sub = read_csv("../input/sample_submission.csv")
# Since the training file is very large, I only use the first 1 million records
train = train_data[:1000000]
# Then we take all the values between the minimum and the maximum for each column

pickup_longitude_min = test.pickup_longitude.min()
pickup_longitude_max = test.pickup_latitude.max()
pickup_latitude_min = test.pickup_latitude.min()
pickup_latitude_max = test.pickup_latitude.max()
dropoff_longitude_min = test.dropoff_longitude.min()
dropoff_longitude_max = test.dropoff_longitude.max()
dropoff_latitude_min = test.dropoff_latitude.min()
dropoff_latitude_max = test.dropoff_latitude.max()

train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 300)]
train = train.loc[(train['pickup_longitude'] > pickup_longitude_min) & (train['pickup_longitude'] < pickup_longitude_max)]
train = train.loc[(train['pickup_latitude'] > pickup_latitude_min) & (train['pickup_latitude'] < pickup_latitude_max)]
train = train.loc[(train['dropoff_longitude'] > dropoff_longitude_min) & (train['dropoff_longitude'] < dropoff_longitude_max)]
train = train.loc[(train['dropoff_latitude'] > dropoff_latitude_min) & (train['dropoff_latitude'] < dropoff_latitude_max)]
# Using the formulas of spherical trigonometry, we find the distance between points

def rasst(value1, value2, value3, value4):

    longitude_1, latitude_1, longitude_2, latitude_2 = value1, value2, value3, value4
    longitude_1, latitude_1, longitude_2, latitude_2 = map(radians, [longitude_1, latitude_1, longitude_2, latitude_2])

    dlongitude = longitude_2 - longitude_1
    dlatitude = latitude_2 - latitude_1

    value = sin(dlatitude/2.0)**2 + cos(latitude_1) * cos(latitude_2) * sin(dlongitude/2.0)**2

    c = 2 * arcsin(sqrt(value))
    km = c * 6367
    return km
# Sort the date by individual columns

train['pickup_datetime'] = to_datetime(train['pickup_datetime'])
train['hour_of_day'] = train.pickup_datetime.dt.hour.astype(float)
train['day'] = train.pickup_datetime.dt.day.astype(float)
train['week'] = train.pickup_datetime.dt.week.astype(float)
train['month'] = train.pickup_datetime.dt.month.astype(float)
train['day_of_year'] = train.pickup_datetime.dt.dayofyear.astype(float)
train['week_of_year'] = train.pickup_datetime.dt.weekofyear.astype(float)
train['passenger_count'] = train['passenger_count'].astype(float)
train['rasst'] = rasst(train['pickup_longitude'], train['pickup_latitude'], train['dropoff_longitude'], train['dropoff_latitude'])

test['pickup_datetime'] = to_datetime(test['pickup_datetime'])
test['hour_of_day'] = test.pickup_datetime.dt.hour.astype(float)
test['day'] = test.pickup_datetime.dt.day.astype(float)
test['week'] = test.pickup_datetime.dt.week.astype(float)
test['month'] = test.pickup_datetime.dt.month.astype(float)
test['day_of_year'] = test.pickup_datetime.dt.dayofyear.astype(float)
test['week_of_year'] = test.pickup_datetime.dt.weekofyear.astype(float)
test['passenger_count'] = test['passenger_count'].astype(float)
test['rasst'] = rasst(test['pickup_longitude'], test['pickup_latitude'], test['dropoff_longitude'], test['dropoff_latitude'])
train_y = train["fare_amount"]
test_key = test['key']
train_x = train.drop(["fare_amount", "key"], axis = 1)
train_x = train_x.drop(['pickup_datetime'], axis = 1)
test = test.drop(['pickup_datetime', 'key'], axis = 1)
train_xgb = xgb.DMatrix(train_x, train_y)
test_xgb = xgb.DMatrix(test)
# Since there is a lot of data, we use the approximation in three-methods

num_round = 2
param = {'max_depth':12, 'eta':0.2,'min_child-weight':2, 'gamma':2, 'booster':'dart', 'three-method':'approx', 'normalize_type':'forest', 'rate_drop':0.3, 'eval_metric':'rmse'}
train = xgb.train(param, train_xgb, num_round)
predict = train.predict(test_xgb, ntree_limit = num_round)
res = DataFrame(test_key)
res.insert(1,'fare_amount', predict)
res.to_csv('result.csv', index=False)
