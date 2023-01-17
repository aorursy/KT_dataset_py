# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from haversine import haversine
from sklearn import model_selection
train_df = pd.read_csv('../input/train.csv', nrows=10000000)
train_df.head(2)

train_df.dtypes
train_df.shape
print(train_df.isnull().sum())
# Given that nulls are lesser in number, let's remove these from the training dataset
print('Old size: %d' %len(train_df))
train_df = train_df.dropna(how='any', axis='rows')
print('New size: %d' %len(train_df))
train_df.describe()
# Fare amount can't be negative
# No. of passenger can neither be 0 not be 208 so dropping anything more than 8 passengers
train_df = train_df[(train_df.fare_amount>0)]
train_df = train_df[(train_df.passenger_count>0) & (train_df.passenger_count<9)]
train_df.shape
# Dropping invalid pick-up and drop locations
train_df = train_df[(train_df.pickup_longitude > -75) & (train_df.pickup_longitude < -72)]
train_df = train_df[(train_df.dropoff_longitude > -75) & (train_df.dropoff_longitude < -72)]
train_df = train_df[(train_df.pickup_latitude > 39) & (train_df.pickup_latitude < 42)]
train_df = train_df[(train_df.dropoff_latitude > 39) & (train_df.dropoff_latitude < 72)]
train_df.shape
def add_distance_feature(df):
    distance= []
    for index, row in df.iterrows():
        distance.append(haversine((row['pickup_latitude'], row['pickup_longitude']),
                                    (row['dropoff_latitude'], row['dropoff_longitude'])))
    df['distance'] = distance
add_distance_feature(train_df)
train_df.head(2)
def add_time_and_day_feature(df):
    df['pickup_datetime']  = pd.to_datetime(df['pickup_datetime'])
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['hour_of_day'] = df['pickup_datetime'].dt.hour
    
add_time_and_day_feature(train_df)
train_df.head(2)
train_df.describe()
train_df = train_df[(train_df.distance > 0.25)]
train_df.describe()
X = train_df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance','day_of_week','hour_of_day']]
y = train_df['fare_amount'].values
 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape, y_train.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor().fit(X_train, y_train)
y_pred = reg.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
test_df = pd.read_csv('../input/test.csv')
add_distance_feature(test_df)
add_time_and_day_feature(test_df)
test_X = test_df[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count','distance','day_of_week','hour_of_day']]

test_y_predictions = reg.predict(test_X)

submission = pd.DataFrame({'key':test_df.key, 'fare_amount':test_y_predictions}, 
                          columns=['key', 'fare_amount'])

submission.to_csv('submission.csv', index=False)

print(os.listdir('.'))
