# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler 

from matplotlib import pyplot 

import seaborn as sns

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', parse_dates=['pickup_datetime', 'dropoff_datetime'], index_col='id')
df_train.head()
df_train.info()
df_train.describe()
def split_datetime(df):

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    return df
def data_preporocessing(df):

    df = split_datetime(df)

    df.loc[df['store_and_fwd_flag'] == 'Y' , 'store_and_fwd_flag'] = 1

    df.loc[df['store_and_fwd_flag'] == 'N' , 'store_and_fwd_flag'] = 0

    filtre = (df['trip_duration'] > 120)  &  (df['passenger_count'] > 0) & (df['trip_duration'] < 3600*2)

    df = df[filtre]

    return df

df_train = data_preporocessing(df_train)
df_train.describe()
TARGET = 'trip_duration'

FEATURES = [

    'passenger_count', 'store_and_fwd_flag','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

    'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour', 'pickup_minute', 'pickup_weekday'

]
X_train = df_train[FEATURES]

y_train = df_train[TARGET]



X_train.shape, y_train.shape
rs = ShuffleSplit(n_splits=4, test_size=.15, train_size=.30)
rf = RandomForestRegressor(n_estimators=12, random_state=42)

losses = -cross_val_score(rf, X_train, y_train, cv=rs, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(l) for l in losses]

np.mean(losses)
rf.fit(X_train, y_train)
df_test = pd.read_csv('../input/test.csv', index_col='id', parse_dates=['pickup_datetime'])

df_test = split_datetime(df_test)

df_test.loc[df_test['store_and_fwd_flag'] == 'Y' , 'store_and_fwd_flag'] = 1

df_test.loc[df_test['store_and_fwd_flag'] == 'N' , 'store_and_fwd_flag'] = 0
X_test = df_test[FEATURES]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission.csv') 

submission['trip_duration'] = y_pred
submission.to_csv('submission.csv', index=False)