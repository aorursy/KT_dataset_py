#imports

%matplotlib inline



import numpy as np

import pandas as pd

import geopy as gp

from geopy.distance import great_circle

import matplotlib.pyplot as plt

from pandas import Series, DataFrame

from sklearn import preprocessing

import matplotlib.pyplot as plt 

from scipy import stats, integrate

import seaborn as sns
train_df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = 1000000)
train_df.info()
train_df.head()
train_df.describe()
def clean_passenger(data):

    data = data.drop(data[data["passenger_count"] > 10].index , axis = 0)

    return data
def clean_location(data):

    data = data.drop( data[(data['pickup_latitude'].isnull()) | (data['pickup_longitude'].isnull()) ].index , axis = 0)

    data = data.drop( data[(data['pickup_latitude'] == 0) | (data['pickup_longitude'] == 0) ].index , axis = 0)

    data = data.drop( data[(data['dropoff_latitude'].isnull()) | (data['dropoff_longitude'].isnull()) ].index , axis = 0)

    data = data.drop( data[(data['dropoff_latitude'] == 0) | (data['dropoff_latitude'] == 0) ].index , axis = 0)

    data = data.drop( (data[ (data['pickup_latitude'] < -90)  | (data['pickup_latitude'] > 90) ]).index , axis = 0  )

    data = data.drop( (data[ (data['dropoff_latitude'] < -90)  | (data['dropoff_latitude'] > 90) ]).index , axis = 0  ) 

    data = data.drop( data[(data['dropoff_latitude'] == data['pickup_latitude']) & (data['dropoff_longitude'] == data['pickup_longitude']) ].index , axis = 0)

    return data
def calc_distance(row):

    coords_1 = (row['pickup_latitude'], row['pickup_longitude'])

    coords_2 = (row['dropoff_latitude'], row['dropoff_longitude'])

    return great_circle(coords_1, coords_2).miles
def calc_tariff_per_mile(data):

    data['distance'] = data.apply(calc_distance , axis=1)

    data['tariff_per_mile'] = (data['fare_amount'] - 2.5) / data['distance']

    return data
def parse_date(data):

    data['pickup_datetime']  = pd.to_datetime(data['pickup_datetime'])

    data['year'] = data['pickup_datetime'].apply(lambda t : pd.to_datetime(t).year)

    data['month'] = data['pickup_datetime'].apply(lambda t : pd.to_datetime(t).month)

    data['week_day'] = data['pickup_datetime'].apply(lambda t : pd.to_datetime(t).weekday)

    data['hour'] = data['pickup_datetime'].apply(lambda t : pd.to_datetime(t).hour)

    return data

    
# bounding_box definition ( west_long , east_long , south_lat , north_lat )

NYC_bounding_box = (-74.26 , -73.71 ,  40.43 , 40.95)

JFK_bounding_box = (-73.86 , -73.75 ,  40.61 , 40.66)

LGA_bounding_box = (-73.91 , -73.82 ,  40.75 , 40.79)

EWR_bounding_box = (-74.19 , -74.15 , 40.67 , 40.70)
def check_boundary_box(boundary_box , longtitude , latitude ):

     if ( (boundary_box[0] < longtitude) & (longtitude < boundary_box[1]) & 

          (boundary_box[2] < latitude) & (latitude < boundary_box[3]) ):      

        return True

     else:

        return False 
def get_trip_type(trip_row):

    if (check_boundary_box(JFK_bounding_box ,  trip_row['pickup_longitude'] , trip_row['pickup_latitude'] ) |

        check_boundary_box(LGA_bounding_box ,  trip_row['pickup_longitude'] , trip_row['pickup_latitude'] ) |

        check_boundary_box(EWR_bounding_box ,  trip_row['pickup_longitude'] , trip_row['pickup_latitude'] )) :

        

        return 'airport'

    

    elif (check_boundary_box(JFK_bounding_box ,  trip_row['dropoff_longitude'] , trip_row['dropoff_latitude'] ) |

          check_boundary_box(LGA_bounding_box ,  trip_row['dropoff_longitude'] , trip_row['dropoff_latitude'] ) |

          check_boundary_box(EWR_bounding_box ,  trip_row['dropoff_longitude'] , trip_row['dropoff_latitude'] )) :

            

        return 'airport'

    

    elif (check_boundary_box(NYC_bounding_box ,  trip_row['pickup_longitude'] , trip_row['pickup_latitude'] ) &

          check_boundary_box(NYC_bounding_box ,  trip_row['dropoff_longitude'] , trip_row['dropoff_latitude'] )):

        

         return 'nyc'

    else:

         return 'out'

      

def classify_nyc_trip(data):

    data['trip_type'] = data.apply(get_trip_type , axis=1)

    return data
train_df = train_df.drop(train_df[train_df['fare_amount'] < 2.5 ].index,axis = 0)

train_df.shape
train_df = clean_passenger(train_df)

train_df.shape
train_df = clean_location(train_df)

train_df.shape
train_df = calc_tariff_per_mile(train_df)

train_df.shape
train_df = parse_date(train_df)

train_df.shape
train_df = classify_nyc_trip(train_df)

train_df.shape
train_df.info()
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Fare Amount Distribution')

sns.distplot(train_df['fare_amount'] , ax = ax[0] )

sns.distplot(np.log(train_df['fare_amount']) , ax = ax[1])
train_df.groupby('passenger_count')['fare_amount'].agg(['mean', 'std', 'count'])
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Conditional Fare Amount Distribution given Passenger Count ')

sns.catplot(x="passenger_count", y="fare_amount" , kind="bar" , data=train_df , ax = ax[0] )

for pc , grouped in train_df.groupby('passenger_count'):

    sns.kdeplot( (grouped['fare_amount']) , label = f'{pc} passengers' , ax = ax[1])



train_df.groupby('year')['fare_amount'].agg(['mean', 'std', 'count'])
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Conditional Fare Amount Distribution given Year ')

sns.catplot(x="year", y="fare_amount" , kind="bar" , data=train_df , ax = ax[0] );

for year , grouped in train_df.groupby('year'):

    sns.kdeplot( np.log(grouped['fare_amount']) , label = f'{year} year' , ax = ax[1]);
train_df.groupby('month')['fare_amount'].agg(['mean', 'std', 'count'])
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Conditional Fare Amount Distribution given month ')

sns.catplot(x="month", y="fare_amount" , kind="bar" , data=train_df , ax = ax[0] );

for month , grouped in train_df.groupby('month'):

    sns.kdeplot( np.log(grouped['fare_amount']) , label = f'{month} month' , ax = ax[1]);
train_df.groupby(['week_day'])['fare_amount'].agg(['mean', 'std', 'count'])
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Conditional Fare Amount Distribution given Week Day ')

sns.catplot(x="week_day", y="fare_amount" , kind="bar" , data=train_df , ax = ax[0] );

for wd , grouped in train_df.groupby('week_day'):

    sns.kdeplot( np.log(grouped['fare_amount']) , label = f'{wd} Week Day' , ax = ax[1]);
train_df.groupby(['hour'])['fare_amount'].agg(['mean', 'std', 'count'])
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Conditional Fare Amount Distribution given Hour')

sns.catplot(x="hour", y="fare_amount" , kind="bar" , data=train_df , ax = ax[0] );

for h , grouped in train_df.groupby('hour'):

    sns.kdeplot( np.log(grouped['fare_amount']) , label = f'{h} hour' , ax = ax[1]);
train_df.groupby(['trip_type'])['fare_amount'].agg(['mean', 'std', 'count'])
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Conditional Fare Amount Distribution given Trip Type')

sns.catplot(x="trip_type", y="fare_amount" , kind="bar" , data=train_df , ax = ax[0] );

for tt , grouped in train_df.groupby('trip_type'):

    sns.kdeplot( np.log(grouped['fare_amount']) , label = f'{tt} Trip Type' , ax = ax[1]);