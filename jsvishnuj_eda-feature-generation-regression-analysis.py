import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from math import radians, asin, sin, cos, sqrt



import warnings

warnings.filterwarnings('ignore')
# loading the dataframe

def load_dataframe(path):

    return pd.read_csv(path, nrows = 100000)

train_dataframe = load_dataframe('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv')
train_dataframe.head()
train_dataframe.shape
train_dataframe['key'].nunique
# Checking for missing values

train_dataframe.isnull().sum()
# dropping missing values

def drop_missing_values(dataframe):

    return dataframe.dropna()

train_dataframe = drop_missing_values(train_dataframe)
train_dataframe.isnull().sum()
def haversine_distance(latitudeA, longitudeA, latitudeB, longitudeB):

    radius_of_earth = 6371.8 #in kilometers

    # converting everything into radians

    latitudeA, longitudeA, latitudeB, longitudeB = radians(latitudeA), radians(longitudeA), radians(latitudeB), radians(longitudeB)

    # finding the difference between the latitudes and longitudes

    latitude_difference = latitudeB - latitudeA

    longitude_difference = longitudeB - longitudeA

    # applyin the haversine formulas

    haversin_latitude = (1 - cos(latitude_difference))/2

    haversin_longitude = (1 - cos(longitude_difference))/2

    haversin_teta = haversin_latitude + (cos(longitudeA) * cos(longitudeB) * haversin_longitude)

    # finding the distance

    distance = 2 * radius_of_earth * asin(sqrt(haversin_teta))

    return distance

# haversine_distance(latitudeA, longitudeA, latitudeB, longitudeB)
def distance_feature(dataframe):

    dataframe['distance'] = haversine_distance(0, 0, 0, 0)

    for i in range(len(dataframe)):

        dataframe['distance'].loc[i] = haversine_distance(dataframe['pickup_latitude'].loc[i],

                                                          dataframe['pickup_longitude'].loc[i],

                                                          dataframe['dropoff_latitude'].loc[i],

                                                          dataframe['dropoff_longitude'].loc[i])

    dataframe = dataframe.drop(['pickup_latitude',

                                            'pickup_longitude',

                                            'dropoff_latitude',

                                            'dropoff_longitude',], axis = 1)

    return dataframe

train_dataframe = distance_feature(train_dataframe)

train_dataframe.head()
train_dataframe['pickup_datetime'].dtype

train_dataframe['pickup_datetime'].head()
def time_taken(dataframe):

    # we will first convert into timestamp and then we will bin this

    dataframe['time_taken'] = pd.to_datetime(dataframe['pickup_datetime']).dt.hour

    dataframe = dataframe.drop(['pickup_datetime'], axis = 1)

    # Converting the time taken into binned values for calculations.

    dataframe['time_taken'] = pd.cut(dataframe['time_taken'],

                                       bins=np.array([-1, 3, 6, 9, 12, 15, 18, 21, 24]),

                                       labels=[0,1,2,3,4,5,6,7])

    return dataframe

train_dataframe = time_taken(train_dataframe)

train_dataframe.head()
# splitting the data into train and testing set

from sklearn.model_selection import train_test_split

X = train_dataframe.drop(['fare_amount', 'key'], axis = 1)

y = train_dataframe['fare_amount']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 101)
# Data is not linear at all and there are just 3 input features and 1 target feature. 

# So I am using Random Forest

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500)
# fitting the model

model.fit(X_train, y_train)
# prediction some sample data for validating the model

predictions = model.predict(X_val)
# checking the performance of the model

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(predictions, y_val)))
test_dataframe = load_dataframe('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')

test_dataframe.head()
test_dataframe = drop_missing_values(test_dataframe)

test_dataframe.isnull().sum()
test_dataframe = distance_feature(test_dataframe)

test_dataframe.head()
test_dataframe = time_taken(test_dataframe)

test_dataframe.head()
# predicting the fare amount

X = test_dataframe.drop(['key'], axis = 1)

fare_amount = model.predict(X)

fare_amount[:10]
submission = pd.DataFrame({'key':test_dataframe['key'], 'fare_amount':fare_amount})

submission.head()
submission.to_csv('submission.csv', index = False)