# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import ShuffleSplit



# Any results you write to the current directory are saved as output.



from sklearn.model_selection import cross_val_score
TARGET = 'trip_duration'
df = pd.read_csv('../input/train.csv');

df = df[df.trip_duration < 3600]

df.head()
df.describe()
# function that calculates the distance between the pickup point and the dropoff 

def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h
# function that calculates in radius the direction of one point according to another one

def bearing_array(lat1, lng1, lat2, lng2): 

    AVG_EARTH_RADIUS = 6371 # in km 

    lng_delta_rad = np.radians(lng2 - lng1) 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    y = np.sin(lng_delta_rad) * np.cos(lat2) 

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad) 

    return np.degrees(np.arctan2(y, x))
# function drops all values of dataset that are too abnormal

def drop_odd_values(dataframe):

    dataframe['passenger_count'] = dataframe.passenger_count.map(lambda x: 1 if x == 0 else x)

    dataframe = dataframe[dataframe.passenger_count <= 6]



    dataframe = dataframe[dataframe.haversine_distance <= 90]

def prep_dataset(dataframe, target='trip_duration'):

    

    # split pickup_datetime

    dataframe['datetime'] = pd.to_datetime(dataframe['pickup_datetime'])

    dataframe['year'] = dataframe['datetime'].dt.year

    dataframe['month'] = dataframe['datetime'].dt.month

    dataframe['day'] = dataframe['datetime'].dt.day

    dataframe['weekday'] = dataframe['datetime'].dt.weekday + 1

    dataframe['hour'] = dataframe['datetime'].dt.hour

    

    # calcul of haversine_distance and bearing of latitudes and longitudes from pickup to dropoff

    dataframe['haversine_distance'] = dataframe.apply(lambda x: haversine_array(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']), axis=1)

    dataframe['bearing'] = dataframe.apply(lambda x: bearing_array(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']),axis=1)



    dataframe.loc[:, 'center_latitude'] = (dataframe['pickup_latitude'].values + dataframe['dropoff_latitude'].values) / 2 

    dataframe.loc[:, 'center_longitude'] = (dataframe['pickup_longitude'].values + dataframe['dropoff_longitude'].values) / 2

    

    # calcul of speed for trips 

    # dataframe['speed'] = (dataframe.haversine_distance/(dataframe.trip_duration/3600))



    # drop_odd_values(dataframe)

    

    #drop odd values



    # all selected_columns

    selected_columns = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'day', 'weekday', 'hour', 'month', 'haversine_distance', 'center_latitude', 'center_longitude', 'bearing']

    return selected_columns
X_train = df[prep_dataset(df)]

y_train = df[TARGET]



df['passenger_count'] = df.passenger_count.map(lambda x: 1 if x == 0 else x)

df = df[df.passenger_count <= 6]

df = df[df.haversine_distance <= 90]

    

X_train.shape, y_train.shape
df.haversine_distance.describe()

# df.head()
cv = ShuffleSplit(n_splits=4, test_size=0.1, train_size=0.2, random_state=0)
rf = RandomForestRegressor()

times = -cross_val_score(rf, X_train, y_train, cv=cv, scoring='neg_mean_squared_log_error')

time = [np.sqrt(l) for l in times]

# np.mean(loses)

time[:5]

np.mean(time)

#times.mean()
rf.fit(X_train, y_train)
df_test = pd.read_csv('../input/test.csv')

df_test.head()
df_test.describe()
X_test = df_test[prep_dataset(df_test)]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission_taxi.csv')