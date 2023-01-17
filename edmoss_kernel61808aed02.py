import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

import os



print(os.listdir("../input"))
def load_data_set(filename):

    df = pd.read_csv('../input/' + filename, parse_dates=['pickup_datetime'])

    df.drop(['store_and_fwd_flag'], axis=1, inplace=True)

    return df
df_train = load_data_set('train.csv')

df_train.head()
df_train.describe()
# Deleting trips without passengers

df_train_has_pass = df_train['passenger_count'] != 0

df_train = df_train[df_train_has_pass]



# Deleting trips that lasted more than 2 hours

df_train_reasonable_trip_duration = df_train['trip_duration'] < (2 * 3600)

df_train = df_train[df_train_reasonable_trip_duration]



df_train.shape
dummies = pd.get_dummies(df_train['vendor_id'])

df_train['vendor_1'] = dummies[1]

df_train['vendor_2'] = dummies[2]



df_train.head()
def create_datetime_columns(df, datetime_column):

    df[datetime_column + '_year'] = df[datetime_column].dt.year

    df[datetime_column + '_month'] = df[datetime_column].dt.month

    df[datetime_column + '_day'] = df[datetime_column].dt.day

    df[datetime_column + '_hours'] = df[datetime_column].dt.hour

    df[datetime_column + '_minutes'] = df[datetime_column].dt.minute

    df[datetime_column + '_seconds'] = df[datetime_column].dt.second

    df[datetime_column + '_weekday'] = df[datetime_column].dt.weekday

    df[datetime_column + '_weekhours'] = df[datetime_column].dt.weekday * 24 + df[datetime_column].dt.hour

    return df
df_train = create_datetime_columns(df_train, 'pickup_datetime')

df_train.head()
TARGET = 'trip_duration'

selected_columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 

           'pickup_datetime_year', 'pickup_datetime_month', 'pickup_datetime_day', 'pickup_datetime_hours', 'pickup_datetime_minutes', 'pickup_datetime_seconds', 'pickup_datetime_weekday', 'pickup_datetime_weekhours',

           'vendor_1', 'vendor_2']



X_train = df_train[selected_columns]

y_train = df_train[TARGET]



df_train.shape, X_train.shape, y_train.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



rfg = RandomForestRegressor()

split = ShuffleSplit(n_splits=3, train_size=.25, test_size=.1, random_state=42)



losses = cross_val_score(rfg, X_train, y_train, cv=split, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(-l) for l in losses]



np.mean(losses)
rfg.fit(X_train, y_train)
df_test = load_data_set('test.csv')

df_test.head()
dummies = pd.get_dummies(df_test['vendor_id'])

df_test['vendor_1'] = dummies[1]

df_test['vendor_2'] = dummies[2]



df_test.head()
df_test = create_datetime_columns(df_test, 'pickup_datetime')

df_test.head()
X_test = df_test[selected_columns]

df_test.shape, X_test.shape
y_pred = rfg.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()

submission.shape, y_pred.shape
submission['trip_duration'] = y_pred

submission.head()
submission.to_csv('submission.csv', index=False)
! ls