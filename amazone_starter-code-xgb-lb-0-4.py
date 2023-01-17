# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from geopy.distance import vincenty
from sklearn.model_selection import train_test_split
import xgboost as xgb
train = pd.read_csv("../input/train.csv", parse_dates=['pickup_datetime', 'dropoff_datetime'])
test = pd.read_csv("../input/test.csv", parse_dates=['pickup_datetime'])
train.head()
def calulate_trip_distance(data):
    data['trip_distance'] = data.apply(lambda x: vincenty((x.pickup_latitude, x.pickup_longitude), (x.dropoff_latitude, x.dropoff_longitude)).km, axis=1)
    return data

train = calulate_trip_distance(train)
test = calulate_trip_distance(test)
def process_date(data):
    data['month'] = data.pickup_datetime.dt.month
    data['week'] = data.pickup_datetime.dt.week
    data['dayofweek'] = data.pickup_datetime.dt.dayofweek
    data['day'] = data.pickup_datetime.dt.day
    data['hour'] = data.pickup_datetime.dt.hour
    data['minute'] = data.pickup_datetime.dt.minute
    return data

train = process_date(train)
test = process_date(test)
flag_encode = {'N': 0, 'Y': 1}
train['store_and_fwd_flag'] = train.store_and_fwd_flag.map(lambda x: flag_encode[x])
test['store_and_fwd_flag'] = test.store_and_fwd_flag.map(lambda x: flag_encode[x])
train.trip_duration[train.trip_duration < 5000].hist(bins=100)
# colums to use
df_columns = ['vendor_id',
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
       'trip_distance', 'month', 'week', 'dayofweek', 'day',
       'hour', 'minute']

# convert trip duration to log space.
X_train, X_val, y_train, y_val = train_test_split(train[df_columns], np.log1p(train['trip_duration'].values), test_size=0.3, random_state=42)

dtrain_all = xgb.DMatrix(train[df_columns].values, np.log1p(train['trip_duration'].values), feature_names=df_columns)
dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dval = xgb.DMatrix(X_val, y_val, feature_names=df_columns)
dtest = xgb.DMatrix(test[df_columns], feature_names=df_columns)
xgb_params = {
    'eta': 0.25,
    'min_child_weight': 10,
    'n_trees': 1000, 
    'max_depth': 10,
    'subsample': 0.95,
    'colsample_bytree': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.log(train['trip_duration'].mean()),
    'silent': 1
}

partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],
                       early_stopping_rounds=20, verbose_eval=20)

num_boost_round = partial_model.best_iteration

#num_boost_round = len(cv_result)
print('num_boost_rounds=' + str(num_boost_round))

# train with all data
model = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round=num_boost_round)
# Prepare submission
subm = pd.DataFrame()
subm['id'] = test.id.values
subm['trip_duration'] = np.exp(model.predict(dtest)) - 1
subm.to_csv('submission_xgb5cv.csv', index=False)
