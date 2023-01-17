import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 1000000)

test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
train['fare_amount'].describe()
train = train.drop(train[train['fare_amount']<0].index, axis=0)

train.shape
train['fare_amount'].describe()
train['fare_amount'].sort_values(ascending=False)
train['passenger_count'].describe()
train[train['passenger_count']>6]
train = train.drop(train[train['passenger_count']==208].index, axis = 0)
train['passenger_count'].describe()
train['pickup_latitude'].describe()
train[train['pickup_latitude']<-90]
train[train['pickup_latitude']>90]
train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)
train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]
train['diff_lat'] = ( train['dropoff_latitude'] - train['pickup_latitude']).abs()

train['diff_long'] = (train['dropoff_longitude'] - train['pickup_longitude'] ).abs()
train.isnull().sum()
train = train.dropna(how = 'any', axis = 'rows')
plot = train.iloc[:2000].plot.scatter('diff_long', 'diff_lat')
train = train[(train.diff_long < 5.0) & (train.diff_lat < 5.0)]
def get_input_matrix(df):

    return np.column_stack((df.diff_long, df.diff_lat, np.ones(len(df))))



train_X = get_input_matrix(train)

train_y = np.array(train['fare_amount'])



print(train_X.shape)

print(train_y.shape)
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)

print(w)
test['diff_lat'] = ( test['dropoff_latitude'] - test['pickup_latitude']).abs()

test['diff_long'] = (test['dropoff_longitude'] - test['pickup_longitude'] ).abs()
test_X = get_input_matrix(test)
test_y = np.matmul(test_X, w).round(decimals = 2)
submission = pd.DataFrame()

submission["key"] = test.key

submission["fare_amount"] = test_y

submission.to_csv('submission.csv', index = False)