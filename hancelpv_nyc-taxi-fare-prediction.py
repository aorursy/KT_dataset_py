# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Set columns to most suitable type to optimize for memory usage
types = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}


# Columns to keep (basically discarding the 'key' column) - thanks to the suggestion by mhviraf
cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
        'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
# train = pd.read_csv('../input/train.csv', nrows=5000000, usecols=cols, dtype=types)
train = pd.read_csv('../input/train.csv', nrows=1000000, usecols=cols, dtype=types)
test = pd.read_csv('../input/test.csv')
samp = pd.read_csv('../input/sample_submission.csv')
train.dropna(how = 'any', axis = 'rows', inplace=True)
train = train[train.fare_amount > 0]
train = train[train['passenger_count'] <= 6]
latitude_mask_pickup = (train.pickup_latitude > -90) &  (train.pickup_latitude < 90)

train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) &  (train.dropoff_latitude < 90)

train = train[latitude_mask_dropoff]
longitude_mask_pickup = (train.pickup_longitude > -180) &  (train.pickup_longitude < 180)

train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) &  (train.dropoff_longitude < 180)

train = train[longitude_mask_dropoff]
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['fare_amount'] , axis=1, inplace=True)
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key
def week_num(day):
    '''
    given the day of the month, return the week number of the month
    '''
    if day <=7 : return 'first'
    if (day > 7) and (day <= 14): return 'second'
    if (day > 14) and (day <= 21): return 'third'
    if (day > 21) and (day <= 28): return 'fourth'
    return 'fifth'
def add_time_features(data):
    data.pickup_datetime =  pd.to_datetime(data.pickup_datetime)
    # Date Features 

    data['hour'] = data.pickup_datetime.dt.hour
    data['day_of_week'] = data.pickup_datetime.dt.weekday_name
    data['day_of_month'] = data.pickup_datetime.dt.day
    data['week_of_month'] = data.day_of_month.map(week_num)
    data['month'] = data.pickup_datetime.dt.month
    data['year'] = data.pickup_datetime.dt.year

    data.hour = data.hour.astype(str)
    data.month = data.month.astype(str)
    data.year = data.year.astype(str)
    data.drop('day_of_month', axis=1, inplace=True)
    
    return data

def add_geo_features(data):
    # Geo Features 
    data['abs_diff_longitude'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['abs_diff_latitude'] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data['manhattan_distance'] = data['abs_diff_longitude'] + data['abs_diff_latitude']

    data['squared_long'] = np.power(data['abs_diff_longitude'],2)
    data['squared_lat'] = np.power(data['abs_diff_latitude'],2)

    data['euclid_distance'] = np.sqrt(data['squared_long'] + data['squared_lat'])
    
    return data
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)
features = ['passenger_count', 'hour', 'day_of_week', 'week_of_month', 'month', 'year', 'abs_diff_longitude', 'abs_diff_latitude', 'manhattan_distance', 'euclid_distance']

all_data = all_data[features]

all_data = pd.get_dummies(all_data)
x = all_data[:n_train]
x_test = all_data[n_train:]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
model_1 = RandomForestRegressor()
model_1.fit(x, y)

model_1_pred = model_1.predict(x_test)
sub_1 = pd.DataFrame()
sub_1['key'] = test_id
sub_1['fare_amount'] = model_1_pred
sub_1.to_csv('submission_rf.csv', index=False)
from keras.models import Sequential
from keras.layers import Dense
num_features = len(x.columns)
model = Sequential()
model.add(Dense(30, input_dim=num_features, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y)
test_pred = model.predict(x_test)
sub = pd.DataFrame()
sub['key'] = test_id
sub['fare_amount'] = test_pred
sub.to_csv('submission_nn.csv', index=False)