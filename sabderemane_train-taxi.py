# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

#print(os.listdir("../input"))

from os import path

# Any results you write to the current directory are saved as output.
# Load the data train

TRAIN_PATH = path.join('..', 'input','train.csv')

df = pd.read_csv(TRAIN_PATH)

df.head()
# Load the data test

TRAIN_PATH = path.join('..', 'input','test.csv')

df_test = pd.read_csv(TRAIN_PATH)

df_test.head()
# Load the data submission

SUBMIT_PATH = path.join('..', 'input','sample_submission.csv')

df_submission = pd.read_csv(SUBMIT_PATH)

df_submission.head()
# Show informations about all data

# df['store_and_fwd_flag'].nunique()

# df.shape

# df.describe()

df.info()
plt.figure(figsize=(15, 4))

plt.hist(x='passenger_count', data=df, orientation='horizontal');
plt.figure(figsize=(20, 5))

df['trip_duration'].hist();
# change pickup_datetime to datetime

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
# encode store_and_fwd_flag column and add column to see if it's the night or not

def preprocess(df):

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday

    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    df['pickup_min'] = df['pickup_datetime'].dt.minute

    df['store_and_fwd_flag_codes'] = df['store_and_fwd_flag'].astype('category').cat.codes

    df['is_night'] = (df['pickup_hour'] > 18) & (df['pickup_hour'] < 7)

    

preprocess(df)

preprocess(df_test)
df.head()
# filter DF by remove rows with 0 passengers and trip duration over 7200 secs

filter_ = (df['passenger_count'] > 0) & (df['trip_duration'] < 7200)

df = df[filter_]

df.shape
TARGET = 'trip_duration'

FEATURES = df.columns.drop(['trip_duration','pickup_datetime', 'dropoff_datetime', 'id', 'store_and_fwd_flag'])
def split_dataset(df, features, target=TARGET):

    X = df[features]

    y = df[target]

    

    return X, y
X_train, y_train = split_dataset(df, features=FEATURES)

X_train.shape, y_train.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, KFold



rf = RandomForestRegressor()

kf = KFold(n_splits=5, random_state=1)
rf.fit(X_train, y_train)
losses = cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(-l) for l in losses]

np.mean(losses)
# Re-instantiate RF for test and fit

rf = RandomForestRegressor()

rf.fit(X_train, y_train)
X_test = df_test[FEATURES]
# test prediction

y_test_pred = rf.predict(X_test)

y_test_pred.mean()
df_submission['trip_duration'] = y_test_pred

df_submission.head()
# comparison

train_mean = df['trip_duration'].mean()

pred_mean = df_submission['trip_duration'].mean()

train_mean, pred_mean
df_submission.to_csv('submission.csv', index=False)