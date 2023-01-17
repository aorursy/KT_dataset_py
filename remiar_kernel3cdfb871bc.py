# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sin, cos, sqrt, atan2, radians



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', index_col='id', parse_dates=['pickup_datetime', 'dropoff_datetime'])

df_train.head(10)
def create_datetime(df):

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_seconde'] = df['pickup_datetime'].dt.minute * 60

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    return df
df_train = create_datetime(df_train)

df_train = df_train[(df_train['pickup_day'] == df_train['dropoff_datetime'].dt.day)]

df_train = df_train[(df_train['passenger_count'] > 0)]

df_train = df_train.loc[(df_train['pickup_latitude'] < 40.91) & (df_train['pickup_latitude'] > 40.55)]

df_train = df_train.loc[ (df_train['pickup_longitude'] < -73.71) & (df_train['pickup_longitude'] > -74.25)]

df_train = df_train.loc[ (df_train['dropoff_latitude'] < 40.91) & (df_train['dropoff_latitude'] > 40.55)]

df_train = df_train.loc[ (df_train['dropoff_longitude'] < -73.71) & (df_train['dropoff_longitude'] > -74.25)]

df_train.head()
TARGET = 'trip_duration'

FEATURES = [

    'passenger_count', 'store_and_fwd_flag','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',

    'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour', 'pickup_minute', 'pickup_seconde', 'pickup_weekday'

]
def create_dataset(df, features, target):

    map_dict = {'N':0, 'Y':1}

    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map(map_dict)

    X = df[features]

    y = df[target]

    

    return X, y


X_train, y_train = create_dataset(df_train, FEATURES, TARGET)





X_train.shape, y_train.shape
from sklearn.model_selection import ShuffleSplit



rs = ShuffleSplit(n_splits=4, test_size=.30, train_size=.30)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=12, random_state=42)

losses = -cross_val_score(rf, X_train, y_train, cv=rs, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(l) for l in losses]

np.mean(losses)
rf.fit(X_train, y_train)
df_test = pd.read_csv('../input/test.csv', index_col='id', parse_dates=['pickup_datetime'])

df_test = create_datetime(df_test)

df_test.head()
X_test = df_test[FEATURES]

map_dict = {'N':0, 'Y':1}

X_test['store_and_fwd_flag'] = X_test['store_and_fwd_flag'].map(map_dict)
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission.csv') 

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)
!ls