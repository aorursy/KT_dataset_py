# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import datetime

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import xgboost as xgb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# print(os.listdir("../input"))

data_path = '../input/'



# Any results you write to the current directory are saved as output.



df_train = pd.read_csv(data_path + 'train.csv')

df_test = pd.read_csv(data_path + 'test.csv')



df_train = df_train.sample(n=5000000)



print(df_train.shape)

print(df_test.shape)
df_train.head()
df_train.dtypes
df_train.describe()
df_train.isnull().sum()
df_train = df_train.dropna()

df_train.isnull().sum()
df_test.dtypes
df_test.describe()
df_test.isnull().sum()
print(df_train.shape)

print(df_test.shape)
df_train = df_train.loc[(df_train['fare_amount'] > 0) & (df_train['fare_amount'] < 200)]

df_train = df_train.loc[(df_train['pickup_longitude'] > -300) & (df_train['pickup_longitude'] < 300)]

df_train = df_train.loc[(df_train['pickup_latitude'] > -300) & (df_train['pickup_latitude'] < 300)]

df_train = df_train.loc[(df_train['dropoff_longitude'] > -300) & (df_train['dropoff_longitude'] < 300)]

df_train = df_train.loc[(df_train['dropoff_latitude'] > -300) & (df_train['dropoff_latitude'] < 300)]

df_train = df_train.loc[df_train['passenger_count'] <= 8]



df_train.head()
df_test = df_test.loc[(df_test['pickup_longitude'] > -300) & (df_test['pickup_longitude'] < 300)]

df_test = df_test.loc[(df_test['pickup_latitude'] > -300) & (df_test['pickup_latitude'] < 300)]

df_test = df_test.loc[(df_test['dropoff_longitude'] > -300) & (df_test['dropoff_longitude'] < 300)]

df_test = df_test.loc[(df_test['dropoff_latitude'] > -300) & (df_test['dropoff_latitude'] < 300)]

df_test = df_test.loc[df_test['passenger_count'] <= 8]



df_test.head()
df_train = df_train.reset_index()

df_train.head()
df_train = df_train.drop(['index'], axis=1)

df_train.head()
ids = df_test['key']

train_Y = df_train['fare_amount']



df_train = df_train.drop(['fare_amount'], axis=1)



print(df_train.shape)

print(df_test.shape)
df_train.head()
df_train['pickup_datetime'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))

df_train['pickup_year'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')

df_train['pickup_month'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')

df_train['pickup_day'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')

df_train['pickup_hour'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')

df_train['pickup_minute'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')

df_train['pickup_second'] = df_train['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')



df_test['pickup_datetime'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))

df_test['pickup_year'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')

df_test['pickup_month'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')

df_test['pickup_day'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')

df_test['pickup_hour'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')

df_test['pickup_minute'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')

df_test['pickup_second'] = df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')
df_train = df_train.drop(['pickup_datetime'], axis=1)

df_test = df_test.drop(['pickup_datetime'], axis=1)
df_train['longitude_diff'] = df_train['dropoff_longitude'] - df_train['pickup_longitude']

df_train['latitude_diff'] = df_train['dropoff_latitude'] - df_train['pickup_latitude']

df_train['distance'] = ((df_train['longitude_diff']**2) + (df_train['latitude_diff']**2))**0.5



df_test['longitude_diff'] = df_test['dropoff_longitude'] - df_test['pickup_longitude']

df_test['latitude_diff'] = df_test['dropoff_latitude'] - df_test['pickup_latitude']

df_test['distance'] = ((df_test['longitude_diff']**2) + (df_test['latitude_diff']**2))**0.5
df_train = df_train.drop(['key'], axis=1)

df_test = df_test.drop(['key'], axis=1)
print(df_train.shape)

print(df_test.shape)
df_train.head()
df_test.head()
# scaler = MinMaxScaler()



# train_X = scaler.fit_transform(df_train)

# test_X = scaler.fit_transform(df_test)
x_train,x_test,y_train,y_test = train_test_split(df_train, train_Y, test_size=0.2, random_state=0)



def xgbmodel(x_train,x_test,y_train,y_test):

    matrix_train = xgb.DMatrix(x_train, label=y_train)

    matrix_test = xgb.DMatrix(x_test, label=y_test)

    model = xgb.train(params={'objective':'reg:linear', 'eval_metric':'rmse'},

                     dtrain=matrix_train,

                     num_boost_round=200,

                     early_stopping_rounds=20,

                     evals=[(matrix_test,'test')])

    return model



myxgbmodel = xgbmodel(x_train,x_test,y_train,y_test)



pred = myxgbmodel.predict(xgb.DMatrix(df_test), ntree_limit=myxgbmodel.best_ntree_limit)
# model = LinearRegression()

# model.fit(df_train, train_Y)

# pred = model.predict(df_test)
pred
submission = pd.DataFrame({'key':ids, 'fare_amount':pred})

submission.to_csv('submission.csv', index=False)