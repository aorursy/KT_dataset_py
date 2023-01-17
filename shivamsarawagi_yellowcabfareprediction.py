import numpy as np # linear algebra

import pandas as pd # CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Graph plot

from scipy import stats

from datetime import datetime

import re

import random

from math import sqrt

import seaborn as sns



# For training ML models to predict data

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/nyc-yellow-taxi-2015-sample-data/train_2015.csv')
df.head()
df.info()
df.isnull().sum()
#plotting data

#plotting fare distribution

def plot_fare_frequency(df):

    df.fare_amount.hist(log=True,bins=50,figsize=(13,5))

    plt.ylabel('frequency')

    plt.xlabel('Fare amount USD')

    plt.title('Fare amount Distribution')

    plt.show()



#plotting pickup lat and lon

def plot_pickup_lat_lon(df):

    plot = df.plot.scatter('pickup_longitude', 'pickup_latitude', figsize = (13,5))



#plotting dropoff lat and lon

def plot_drop_lat_lon(df):

    plot = df.plot.scatter('dropoff_longitude', 'dropoff_latitude', figsize = (13,5))



print("\n-------------Visualizing Data-------------\n")

print(df.describe())

plot_fare_frequency(df)

plot_pickup_lat_lon(df)

plot_drop_lat_lon(df)
#cleaning data

#remove na rows

def drop_na_rows(df):

    print(df.isnull().sum())

    return df.dropna(how = 'any', axis = 'rows')



#remove negative fare

def remove_negative_fare_amount(df):

    return df[(df['fare_amount'] > 0)]





#removing rows outside newyork city

def remove_rows_latitude_longitude_not_in_range(df):

    MAX_LONGITUDE = -72.586532

    MIN_LONGITUDE = -74.663242



    MAX_LATITUDE = 41.959555

    MIN_LATITUDE = 40.168973,



    df = df[(MIN_LONGITUDE <= df.dropoff_longitude) &(df.dropoff_longitude <= MAX_LONGITUDE)

            & (MIN_LONGITUDE <= df.pickup_longitude) &(df.pickup_longitude <= MAX_LONGITUDE)

            & (MIN_LATITUDE <= df.dropoff_latitude) & (df.dropoff_latitude <= MAX_LATITUDE)

            & (MIN_LATITUDE <= df.pickup_latitude) & (df.pickup_latitude <= MAX_LATITUDE)]

    

    return df



# Removing rows with passenger count not in range

def remove_rows_passenger_count_not_in_range(df):

    return df[(df.passenger_count > 0) & (df.passenger_count < 7)]
#Actual Cleaning begins here

def clean_data(df):

    

    old_size = len(df)

    df = drop_na_rows(df)

    new_size = len(df)

    print('Dropped NA rows. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    

    old_size = new_size

    df = remove_negative_fare_amount(df)

    new_size = len(df)

    print('Cleaned data for fare amount. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    

    old_size = new_size

    df = remove_rows_latitude_longitude_not_in_range(df)

    new_size = len(df)

    print('Cleaned data for latitude longitude. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    

    old_size = new_size

    df = remove_rows_passenger_count_not_in_range(df)

    new_size = len(df)

    print('Cleaned data for passenger count. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    return df



df = clean_data(df)
fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

plt.xlabel('pickup_longitude')

plt.ylabel('pickup_latitude')

plt.title('Pickup Longitude vs Latitude Plot')

ax.scatter(df['pickup_longitude'], df['pickup_latitude'], s = 0.007, alpha = 1)

plt.show()



fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

plt.xlabel('dropoff_longitude')

plt.ylabel('dropoff_latitude')

plt.title('Dropoff Longitude vs Latitude Plot')

ax.scatter(df['dropoff_longitude'], df['dropoff_latitude'], s = 0.007, alpha = 1)

plt.show()
#calculate haversine distance

def haversine_distance(lon1,lat1,lon2,lat2):

    p = 0.01745329251

    a = 0.5-np.cos((lat2-lat1)*p)/2+np.cos(lat1*p)*np.cos(lat2*p)*(1-np.cos((lon2-lon1)*p))/2

    return 12742 * np.arcsin(np.sqrt(a))



def add_eucledian_distance_feature(df):

    df['eucledian_distance'] = haversine_distance(df.dropoff_longitude, df.dropoff_latitude, 

                                                  df.pickup_longitude, df.pickup_latitude)



date_regex = re.compile('([0-9]{4})-([0-9]{2})-([0-9]{2}) ([0-9]{2}):([0-9]{2}):([0-9]{2})(.*)')  



def pickup_datetime_parser(x):

    m = date_regex.search(x)

    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)))



# Adding new feature of related to time

def add_time_feature(df):

    df['pickup_datetime_obj'] = df.tpep_pickup_datetime.apply(pickup_datetime_parser)

    df['year'] = df.pickup_datetime_obj.apply(lambda x: x.year)

    df['month'] = df.pickup_datetime_obj.apply(lambda x: x.month)

    df['weekday'] = df.pickup_datetime_obj.apply(lambda x: x.isoweekday())

    df['hour'] = df.pickup_datetime_obj.apply(lambda x: x.hour)

    df['minute'] = df.pickup_datetime_obj.apply(lambda x: (x.hour * 60) + x.minute)

    

    return df





def add_travel_vector(df):

    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()

    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

    return df



#adding new features

def transform_data(df):

    old_size = len(df)

    add_eucledian_distance_feature(df)

    new_size = len(df)

    print('Added eucledian distance feature. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    

    old_size = new_size

    df = add_time_feature(df)

    new_size = len(df)

    print('Added time related features. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    

    old_size = new_size

    df = add_travel_vector(df)

    new_size = len(df)

    print('Added travel vector features. New size: %d Total Rows removed: %d' % (new_size, old_size - new_size))

    

    return df



df = transform_data(df)
# Checks if the given lat lon is in Manhattan

def is_in_manhattan(lon, lat):

    BB = (-74.025, -73.925, 40.7, 40.8)

    

    return ((lon >= BB[0]) & (lon <= BB[1]) & (lat >= BB[2]) & (lat <= BB[3]))



# Checks if the given lat lon is in airport region.

# Currently, only considering JFK for the airport.

def is_near_airport(lon, lat):

    MAX_RADIUS = 2

    

    # JFK coordinates

    AIRPORT_COORDINATES = [(40.645112, -73.785524)]



    within_radius = False

    for t in AIRPORT_COORDINATES:

        within_radius = within_radius or (haversine_distance(lon, lat, t[1], t[0]) <= MAX_RADIUS)

        

    return within_radius



# Checks if the trip is an airport trip or not.

def is_airport_trip(x):

    return ((is_near_airport(x.dropoff_longitude, x.dropoff_latitude) \

                and is_in_manhattan(x.pickup_longitude, x.pickup_latitude))

            or (is_near_airport(x.pickup_longitude, x.pickup_latitude) \

                and is_in_manhattan(x.dropoff_longitude, x.dropoff_latitude)))



def airport_trips(df):    

    return df[df.apply(is_airport_trip, axis = 1)]



train_df_airport = airport_trips(df)



print("Total number of airport trips: %d" % len(train_df_airport))



print("\n-----------Visualizing airport trips------------\n")



fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (7,7))

plt.xlim(12.5,20)

plt.xlabel('eucledian_distance')

plt.ylabel('fare_amount')

plt.title('Fare amount vs Eucledian distance for Airport Trips')

ax.scatter(train_df_airport['eucledian_distance'], train_df_airport['fare_amount'])

plt.show()



fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize = (12,10))

plt.ylim(40.6, 40.9)

plt.xlim(-74.1,-73.7)

plt.xlabel('pickup_longitude/dropoff_longitude')

plt.ylabel('pickup_latitude/dropoff_latitude')

plt.title('Pickup/Dropoff Longitude vs Latitude Plot for JFK-Manhattan')

ax.scatter(train_df_airport['pickup_longitude'], train_df_airport['pickup_latitude'], s = 0.1, alpha = 1, label = 'pickup')

ax.scatter(train_df_airport['dropoff_longitude'], train_df_airport['dropoff_latitude'], s = 0.1, alpha = 1, label = 'dropoff')

ax.legend()

plt.show()
df = df.drop(['pickup_datetime_obj'],axis=1)
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

df['trip_time'] =  df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
df = df.drop(['year'],axis=1)
plt.figure(figsize= (10, 5))

sns.heatmap(df.corr())
df['trip_time'] = (df['trip_time'].dt.components['hours'] * 60 ) + df['trip_time'].dt.components['minutes']
plt.figure(figsize= (10, 5))

sns.heatmap(df.corr())
df.columns
df.drop(columns=['extra','improvement_surcharge','mta_tax','tip_amount','total_amount','tolls_amount','tpep_dropoff_datetime','tpep_pickup_datetime','dropoff_latitude','dropoff_longitude','pickup_latitude','pickup_longitude'],inplace=True)
df.head()
df['RateCodeID'].value_counts()
df['VendorID'].value_counts()
df['payment_type'].value_counts()
plt.figure(figsize= (10, 5))

sns.heatmap(df.corr())
one_hot = pd.get_dummies(df['store_and_fwd_flag'])

df = df.drop('store_and_fwd_flag',axis = 1)

df = df.join(one_hot)
df.head()
train_df, test_df = train_test_split(df, test_size = 0.25)
def test_model(regr, train_df, test_df):

    X = train_df.drop(['fare_amount'],axis=1)

    y = train_df['fare_amount'] 

    regr.fit(X, y)



    y_pred = regr.predict(X)

    err = sqrt(mean_squared_error(y, y_pred))

    regr.score(X,y)

    print("Training Root Mean squared error: %.4f" % (err))



    # Validation Testing

    X_test = test_df.iloc[:, 1:]

    y_test = test_df.iloc[:, 0]   

    y_test_pred = regr.predict(X_test)



    test_err = sqrt(mean_squared_error(y_test, y_test_pred))

    print("Validation Root Mean squared error: %.4f" % (test_err))
rf_regr = RandomForestRegressor(random_state=0, n_jobs = -1, n_estimators = 10, oob_score = True)

test_model(rf_regr, train_df, test_df)