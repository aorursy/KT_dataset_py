import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

import os





print(os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv", parse_dates=['pickup_datetime'])
df_train['pk_year'] = df_train['pickup_datetime'].dt.year

df_train['pk_month'] = df_train['pickup_datetime'].dt.month

df_train['pk_day'] = df_train['pickup_datetime'].dt.day

df_train['pk_hour'] = df_train['pickup_datetime'].dt.hour

df_train['pk_minute'] = df_train['pickup_datetime'].dt.minute

df_train['pk_second'] = df_train['pickup_datetime'].dt.minute*60
rs = ShuffleSplit(n_splits=3, test_size=0.05, train_size=.1,random_state=0)
selected_columns = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude',

                    'dropoff_longitude','dropoff_latitude','pk_year',

                    'pk_month','pk_day','pk_hour','pk_minute','pk_second',]

X_train = df_train[selected_columns]

y_train = df_train['trip_duration']

X_train.shape, y_train.shape
rf = RandomForestRegressor()

losses = -cross_val_score(rf, X_train, y_train, cv=rs, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(-l) for l in losses]

np.mean(losses)
rf.fit(X_train,y_train)
df_test = pd.read_csv("../input/test.csv", parse_dates=['pickup_datetime'])
df_test['pk_year'] = df_test['pickup_datetime'].dt.year

df_test['pk_month'] = df_test['pickup_datetime'].dt.month

df_test['pk_day'] = df_test['pickup_datetime'].dt.day

df_test['pk_hour'] = df_test['pickup_datetime'].dt.hour

df_test['pk_minute'] = df_test['pickup_datetime'].dt.minute

df_test['pk_second'] = df_test['pickup_datetime'].dt.minute*60

selected_columns = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude',

                    'dropoff_longitude','dropoff_latitude','pk_year',

                    'pk_month','pk_day','pk_hour','pk_minute','pk_second',]

X_test = df_test[selected_columns]

X_test.shape
y_pred=rf.predict(X_test)

math.sqrt(-cross_val_score(rf, X_test, y_pred, cv = rs, scoring='neg_mean_squared_log_error').mean())
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()

submission.shape, y_pred.shape
submission['trip_duration'] = y_pred

submission.head()
submission.to_csv('submission.csv', index=False)
! ls