# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import xgboost as xgb
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", nrows = 5_000_000)
# train = pd.read_csv("../input/train.csv", nrows = 100000)
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.describe()
# train dataset have negative fare_amount and 0 passenger_count
# check same thing for test dataset
test.describe()
# check for null values of train
train.isnull().sum()
# check for null values of test
test.isnull().sum()
# when date time info is not used the error is very high
def handle_date(df):    
    df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')    
    df['hour_of_day'] = df.pickup_datetime.dt.hour
    df['week'] = df.pickup_datetime.dt.week
    df['month'] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df['day_of_year'] = df.pickup_datetime.dt.dayofyear
    df['week_of_year'] = df.pickup_datetime.dt.weekofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day
    df = df.drop('pickup_datetime', axis = 1)
    return df

train = handle_date(train)
test = handle_date(test)
def handle_distance(df):
    df['longitude_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['latitude_distance'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])
    df['distance_travelled'] = (df['longitude_distance'] ** 2 + df['latitude_distance'] ** 2) ** .5
    df = df.drop(['longitude_distance','latitude_distance'], axis=1)
    return df

train = handle_distance(train)
test = handle_distance(test)
train.describe()
def clean_up_train(train):                                               
    train = train.dropna()
    train = train[train['fare_amount'] > 0]
    train = train[train['passenger_count'] > 0]
    train = train[train['passenger_count'] < 7]
    return train

train = clean_up_train(train)
# must better now
train.describe()
def get_samples_output(train):
    return (train[test.drop('key', axis=1).columns], train['fare_amount'])
from sklearn.model_selection import train_test_split

samples_train, samples_label = get_samples_output(train.drop('key', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(samples_train, samples_label, test_size=0.3, random_state=0)
## used for selecting algorithms but to mess to show
# from sklearn.ensemble import RandomForestRegressor
# submission = pd.DataFrame(
#     {'key': test.key, 'fare_amount': RandomForestRegressor(n_estimators=10, max_depth=2).fit(samples_train, samples_label).predict(test.drop('key', axis=1))},
#     columns = ['key', 'fare_amount'])
# submission.to_csv('submission.csv', index=False)
# submission.head(20)
model = xgb.XGBRegressor(max_depth=2, n_estimators=50, silent=False)
model.fit(samples_train, samples_label)

submission = pd.DataFrame(
    {'key': test.key, 'fare_amount': model.predict(test.drop('key', axis=1))},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index=False)
submission.head(20)