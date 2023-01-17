# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import geopy.distance

# Any results you write to the current directory are saved as output.
df_train= pd.read_csv("../input/train.csv", nrows=500000,low_memory=True)
df_test = pd.read_csv("../input/test.csv")
df_train.info()
df_train.head(5)
df_train['key2'] = pd.to_datetime(df_train['key'], errors='coerce')
df_train.info()
df_test['key2'] = pd.to_datetime(df_test['key'], errors='coerce')
df_train.drop('pickup_datetime', axis=1, inplace=True)
df_test.drop('pickup_datetime', axis=1, inplace=True)
df_train.isnull().sum()
df_train.dropna(axis=0, inplace=True)
df_train['fare_amount'].plot(kind='box')
df_train.describe()
print("% of fares above 25$ - {:0.2f}".format(df_train[df_train['fare_amount'] > 25]['key'].count()*100/df_train['key'].count()))
df_train = df_train[~(df_train['fare_amount'] > 25)]
df_train['fare_amount'].plot(kind='box')
df_train = df_train[~(df_train['fare_amount'] < 0)]
df_train['fare_amount'].plot(kind='box')
df_train.info()
df_train['passenger_count'].plot(kind='box')
print("% of passengers above 6 - {:0.2f}".format(df_train[df_train['passenger_count'] > 6]['key'].count()*100/df_train['key'].count()))
df_train = df_train[~(df_train['passenger_count'] > 6)]
df_train['passenger_count'].plot(kind='box')
print("Count of invalid pickup latitude", df_train[(df_train['pickup_latitude'] > 90) | (df_train['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", df_train[(df_train['dropoff_latitude'] > 90) | (df_train['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", df_train[(df_train['pickup_longitude'] > 180) | (df_train['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", df_train[(df_train['dropoff_longitude'] > 180) | (df_train['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())
print("Count of invalid pickup latitude", df_test[(df_test['pickup_latitude'] > 90) | (df_test['pickup_latitude'] < -90) ]['pickup_latitude'].count())
print("Count of invalid dropoff latitude", df_test[(df_test['dropoff_latitude'] > 90) | (df_test['dropoff_latitude'] < -90) ]['dropoff_latitude'].count())
print("Count of invalid pickup longitude", df_test[(df_test['pickup_longitude'] > 180) | (df_test['pickup_longitude'] < -180) ]['pickup_longitude'].count())
print("Count of invalid dropoff longitude", df_test[(df_test['dropoff_longitude'] > 180) | (df_test['dropoff_longitude'] < -180) ]['dropoff_longitude'].count())
df_train = df_train[~((df_train['pickup_latitude'] > 90) | (df_train['pickup_latitude'] < -90))]
df_train = df_train[~((df_train['dropoff_latitude'] > 90) | (df_train['dropoff_latitude'] < -90))]
df_train = df_train[~((df_train['pickup_longitude'] > 180) | (df_train['pickup_longitude'] < -180))]
df_train = df_train[~((df_train['dropoff_longitude'] > 180) | (df_train['dropoff_longitude'] < -180))]
df_train['distance']=df_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)
df_train.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)
df_test['distance']=df_test[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].apply(lambda x: 
                                                                                                geopy.distance.VincentyDistance((x['pickup_latitude'],x['pickup_longitude']),
                                                                                                                               (x['dropoff_latitude'],x['dropoff_longitude'])).km,axis=1)
df_test.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],axis=1, inplace=True)
print("% of trips above 25 KM - {:0.2f}".format(df_train[df_train['distance'] > 25]['key'].count()*100/df_train.count()['key']))
df_train = df_train[~(df_train['distance'] > 25)]
df_train['distance'].plot(kind='box')
df_train['year'] = df_train['key2'].apply(lambda x : x.year)
df_train['month'] = df_train['key2'].apply(lambda x : x.month)
df_train['day'] = df_train['key2'].apply(lambda x : x.day)
df_train['day of week'] = df_train['key2'].apply(lambda x : x.weekday())
df_train['hour'] = df_train['key2'].apply(lambda x : x.hour)
df_train.head(10)
df_test['year'] = df_test['key2'].apply(lambda x : x.year)
df_test['month'] = df_test['key2'].apply(lambda x : x.month)
df_test['day'] = df_test['key2'].apply(lambda x : x.day)
df_test['day of week'] = df_test['key2'].apply(lambda x : x.weekday())
df_test['hour'] = df_test['key2'].apply(lambda x : x.hour)
Y = df_train['fare_amount']
df_train.drop('fare_amount',axis=1, inplace=True)
df_train.head(5)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train[['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour']], Y, test_size=0.1)
import xgboost as xg
xgbr = xg.XGBRegressor(learning_rate=.35,max_depth=4,n_estimators=150, booster='dart')
xgbr.fit(X_train, y_train)
pred = xgbr.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, pred)
xgbr.feature_importances_
pred = xgbr.predict(df_test[['passenger_count', 'distance', 'year', 'month', 'day', 'day of week', 'hour']])
submission = pd.DataFrame({"key": df_test['key'], "fare_amount": pred})
submission.to_csv('submission.csv',index=False)
