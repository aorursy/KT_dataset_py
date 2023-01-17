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

  #  data = data.drop( data[(data['dropoff_latitude'] == data['pickup_latitude']) & (data['dropoff_longitude'] == data['pickup_longitude']) ].index , axis = 0)

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
def estimate_conditional_pdf(data,x,xLabel):

    y = 'fare_amount'

    print (data.groupby([x])[y].agg(['mean', 'std', 'count']))

    fig, ax = plt.subplots(1,2 , figsize=(18,5))

    fig.suptitle('Conditional Fare Amount Distribution given '+ xLabel)

    sns.barplot(x = x , y = y ,  data = data , ax = ax[0] )

    for  xgrp , grp_mean  in data.groupby(x):

        sns.kdeplot( np.log(grp_mean[y]) , label = f'{xgrp} '+xLabel , ax = ax[1])
def partition_time(data):

    bins = [ 0  , 6 , 12  , 16 , 20 , 24]

    labels = ['overnight' , 'morning', 'afternoon', 'rush_hour' , 'evening']

    data['time_group'] = pd.cut(data['hour'], bins, labels=labels)

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
train_df = partition_time(train_df)

train_df.shape
train_df = classify_nyc_trip(train_df)

train_df.shape
train_df.info()
fig, ax = plt.subplots(1,2 , figsize=(18,5))

fig.suptitle('Fare Amount Distribution')

sns.distplot(train_df['fare_amount'] , ax = ax[0] )

sns.distplot(np.log(train_df['fare_amount']) , ax = ax[1])
estimate_conditional_pdf(train_df, 'passenger_count' , 'Passenger Count')
estimate_conditional_pdf(train_df, 'year' , 'Year')
estimate_conditional_pdf(train_df, 'month' , 'Month')
estimate_conditional_pdf(train_df, 'week_day' , 'Week Day')
estimate_conditional_pdf(train_df, 'hour' , 'Hour')
train_df = partition_time(train_df)
estimate_conditional_pdf(train_df, 'time_group' , 'Time Group')
estimate_conditional_pdf(train_df,'trip_type','Trip Type')
train_df.info()
train_df = pd.get_dummies( train_df, columns = ['trip_type'] )

train_df = pd.get_dummies( train_df, columns = ['time_group'] )
train_df.info()
from sklearn.model_selection import train_test_split

features = [ 'passenger_count' ,  'distance' ,  'year' ,  'week_day' ,   'trip_type_airport' , 'trip_type_nyc' , 'trip_type_out' , 'time_group_overnight' , 'time_group_morning' , 'time_group_afternoon' , 'time_group_rush_hour' , 'time_group_evening' ]

X_train, X_test, Y_train, Y_test = train_test_split( train_df[features] , train_df[['fare_amount']] , test_size=0.25 , random_state=1)

models = {}
# Add linear regression model

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



models['linear_model'] = Pipeline((

        ("standard_scaler", StandardScaler()),

        ("linear_reg", LinearRegression()),

    ))
# Add linear model with polynomial features. Use Ridge for L2 regularization

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge



models['polynomial'] = Pipeline((

        ("standard_scaler", StandardScaler()),    

        ("poly_features", PolynomialFeatures(degree=2)),

        ("ridge", Ridge()),

    ))
from sklearn.ensemble import GradientBoostingRegressor



models['gradient_boosting_n10'] = GradientBoostingRegressor(max_depth=2, n_estimators=10, learning_rate=1.0)

models['gradient_boosting_n100'] = GradientBoostingRegressor(max_depth=2, n_estimators=100, learning_rate=0.2)
# Add RandomForestRegressor with several different parameters

from sklearn.ensemble import RandomForestRegressor



models['random_forest_regressor_n10'] = RandomForestRegressor(n_estimators=10, max_depth=10, min_samples_leaf=10)

models['random_forest_regressor_n100'] = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10)
from sklearn import metrics

for name, model in models.items():

    model.fit(X_train,Y_train)

    print('\n... model name {} ...'.format(name))

    Y_predict = model.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_predict))

    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_predict))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))