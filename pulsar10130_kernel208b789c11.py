import numpy as np 

import pandas as pd 

import math

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_train.info()

#object type will need processing
df_train.describe()

#0 passenger!?

#3.526282e+06/3600 ~= 979 522 778 hours what the duck!?

#Target's mean is 9.594923e+02
sns.boxplot(x=df_train['trip_duration'])
sns.boxplot(x=df_train['passenger_count'])
sns.boxplot(x=df_train['pickup_longitude'])

#sns.boxplot(x=df_train['pickup_latitude'])
#Let's put away the data with more than 2h of trip duration ---> the boxplot(twist) after first filter point towards <2000, let'say 1 hour(3600)

df_train = df_train[(df_train.trip_duration < 3600)]

#And with 0 passenger ---> see boxplot

df_train = df_train[(df_train.passenger_count > 0)]

df_train = df_train[(df_train.passenger_count < 5)]

#I spy with my little eyes some weird coordinates

df_train = df_train[(df_train.pickup_latitude < 41)]

df_train = df_train[(df_train.pickup_latitude > 40.5)]

df_train = df_train[(df_train.pickup_longitude > -74.5)]

df_train = df_train[(df_train.pickup_longitude < -73.5)]





df_train.describe()
#df_train['vendor_id'].nunique()

#df_train['passenger_count'].nunique()

#df_train['store_and_fwd_flag'].nunique()
#Let's create a function to avoid repetition with test.csv

def process_dataset(df):

    #Datetime casting and parsing

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['hour_in_month'] = ((df['pickup_day'] * 24) + df['pickup_hour'])    

    

    #Let's calculate the distance via long/lat of pickup and dropoff via old pal' Pythagore

    df['lat_delta'] = np.abs(df['pickup_latitude'] - df['dropoff_latitude'])

    df['long_delta'] = np.abs(df['pickup_longitude'] - df['dropoff_longitude'])

    df['straight_distance'] = np.sqrt(np.square(df['lat_delta']) + np.square(df['long_delta']))

    #another idea : the streets of nyc are more or less perpendicular so maybe lat_delta + long_delta is a better distance evaluation

    df['driving_distance'] = df['lat_delta'] + df['long_delta']

    

    #Get dummies

    df['store_and_fwd_flag'] = pd.get_dummies(df['store_and_fwd_flag'], drop_first=True)

    df['vendor_id'] = pd.get_dummies(df['vendor_id'], drop_first=True)

    

    #selecting columns

    selected_columns = ['pickup_longitude', 'pickup_latitude', 

                        'dropoff_longitude', 'dropoff_latitude', 

                        'pickup_year', 'pickup_month', 'pickup_hour', 

                        'pickup_dayofweek', 'pickup_day', 'hour_in_month', 

                        'straight_distance', 'driving_distance', 'store_and_fwd_flag', 'passenger_count', 'vendor_id']    

    X = df[selected_columns]

    

    return X
X_train = process_dataset(df_train)

y = df_train['trip_duration']
X_train.info()
rf = RandomForestRegressor()

#too many data for a cross validation, let's split!

rs = ShuffleSplit(n_splits=3, train_size =.20, test_size=.25, random_state=0)

losses = -cross_val_score(rf, X_train, y, cv = rs, scoring = 'neg_mean_squared_log_error')

losses.mean()

rf.fit(X_train, y)
df_test = pd.read_csv('../input/test.csv')

X_test = process_dataset(df_test)

X_test.head()

#Of course nether dropoff_datetime nor trip duration
y_pred = rf.predict(X_test)

math.sqrt(-cross_val_score(rf, X_train, y, cv = rs, scoring='neg_mean_squared_log_error').mean())
submission = pd.read_csv('../input/sample_submission.csv') 

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)
!ls