import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

#import sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')
df_train.head()
df_test.head(1)
df_train.describe()
df_train.info()
df_train['passenger_count'].value_counts()
#Visualise the minimum and maximum trip duration

df_train.sort_values(by='trip_duration', ascending=False)
#Extreme trip durations and trip with zero passenger

df_train = df_train = df_train[(df_train['passenger_count'] > 0) & (df_train['trip_duration'] < 8000)]
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])

#df_train['year'] = df_train['pickup_datetime'].dt.year

df_train['month'] = df_train['pickup_datetime'].dt.month

df_train['day'] = df_train['pickup_datetime'].dt.day

df_train['hour'] = df_train['pickup_datetime'].dt.hour

df_train['minute'] = df_train['pickup_datetime'].dt.minute

df_train['second'] = df_train['pickup_datetime'].dt.second

df_train['weekday'] = df_train['pickup_datetime'].dt.weekday

df_train.head(1)
df_train.loc[df_train['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0

df_train.loc[df_train['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1

df_train.head(2)
def split_dataset(df, features, target='trip_duration'):

    X = df[features]

    y = df[target]

    return X, y
# predictor features

selected_columns = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 

                    'month', 'day', 'hour', 'minute', 'second', 'weekday', 'store_and_fwd_flag']

X_train, y_train = split_dataset(df_train, selected_columns)

# X_train = df_train[selected_columns]

# y_train = df_train['trip_duration']

X_train.shape, y_train.shape
rf = RandomForestRegressor()

score = -cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_log_error')

score.mean()
rf.fit(X_train, y_train)
rf.feature_importances_
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

#df_test['year'] = df_test['pickup_datetime'].dt.year

df_test['month'] = df_test['pickup_datetime'].dt.month

df_test['day'] = df_test['pickup_datetime'].dt.day

df_test['hour'] = df_test['pickup_datetime'].dt.hour

df_test['minute'] = df_test['pickup_datetime'].dt.minute

df_test['second'] = df_test['pickup_datetime'].dt.second

df_test['weekday'] = df_test['pickup_datetime'].dt.weekday

df_test.loc[df_test['store_and_fwd_flag'] == 'N', 'store_and_fwd_flag'] = 0

df_test.loc[df_test['store_and_fwd_flag'] == 'Y', 'store_and_fwd_flag'] = 1

df_test.head()
X_test = df_test[selected_columns]
y_test_pred = rf.predict(X_test)

y_test_pred.mean()
submission.head()
submission['trip_duration'] = y_test_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)
!ls