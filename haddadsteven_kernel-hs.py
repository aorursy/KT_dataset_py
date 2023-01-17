import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor

import os

sns.set()

print(os.listdir("../input"))

import datetime as dt



from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', parse_dates=['pickup_datetime','dropoff_datetime'])

df_train.head()
df_train.describe()
sns.countplot(x=df_train['trip_duration']<3000, hue=df_train['passenger_count'])
sns.boxplot(x=df_train['trip_duration']>7200, y=df_train["store_and_fwd_flag"])

sns.countplot(x=df_train['trip_duration']<3000, hue=df_train['vendor_id'])
df_train[(df_train.trip_duration > 7000)].head()
df_train = df_train[(df_train.trip_duration < 7000)]

df_train = df_train[(df_train.passenger_count != 0)]

df_train.head()
def extract_datetime(df):

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_seconde'] = df['pickup_datetime'].dt.minute * 60

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    return df





selected_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

                    'dropoff_latitude','pickup_year','pickup_month','pickup_day','pickup_hour','pickup_minute',

                    'pickup_seconde','pickup_weekday']





target='trip_duration'



extract_datetime(df_train)


X_train = df_train[selected_columns]

y_train = df_train[target]

X_train.shape, y_train.shape
rs = ShuffleSplit(n_splits=3, test_size=.12, train_size=.25)
rf = RandomForestRegressor(n_estimators=12, random_state=42)

losses = -cross_val_score(rf, X_train, y_train, cv=rs, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(l) for l in losses]

np.mean(losses)
rf.fit(X_train, y_train)
df_test = pd.read_csv('../input/test.csv',parse_dates=['pickup_datetime'])

extract_datetime(df_test)

df_test.head()
X_test = df_test[selected_columns]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission.csv') 

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)