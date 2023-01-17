import pandas as pd
import numpy as np
from math import cos, asin, sqrt, radians, sin, atan

def sin_sq_half_angle(r):
    return (1 - cos(r)) / 2

def haversine_formula(d_lat, p_lat, d_lon, p_lon):
    earth_radius = 6371 #km
    
    # Application of the Haversine formula using
    # https://en.wikipedia.org/wiki/Haversine_formula
    # Calculating for d
    
    a = sin_sq_half_angle(d_lat - p_lat) + (cos(p_lat) * cos(d_lat) * sin_sq_half_angle(d_lon - p_lon))
    return earth_radius * 2 * asin(sqrt(a))

def haversine_distance(row):
    p_lon = radians(row['pickup_longitude'])
    d_lon = radians(row['dropoff_longitude'])
    p_lat = radians(row['pickup_latitude'])
    d_lat = radians(row['dropoff_latitude'])
    return haversine_formula(d_lat, p_lat, d_lon, p_lon)

# https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration/notebook
# Use bounding box to discard any outliers
# define bounding box
BB = (-75, -73, 40, 41.5)

# this function will be used with the test set below
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            
def transform_dataset(df):
    # Filter any fare amounts above 0 or anything huge ( > $200 USD)
    df = df[(df.fare_amount >= 2.50) & (df.fare_amount < 200) ]
    
    #Filter journeys with no passengers
    df = df[df.passenger_count > 0]
    
    # Calculate haversine distance
    distances = df.filter(regex='longitude|latitude')
    df['distance'] = distances.apply(haversine_distance, axis=1)
    
    #Remove anything with no distance information
    df = df[df.distance > 0]
    distances = distances.loc[df.index.astype(int)]
    
    df = df[['distance', 'fare_amount']]
    return df, distances

def process_data(filename, num_rows=100):
    df = pd.read_csv(filename, nrows=num_rows).dropna()
    df = df[select_within_boundingbox(df, BB)]
    return transform_dataset(df)


df, df_coordinates = process_data('../input/new-york-city-taxi-fare-prediction/train.csv', num_rows=200000)
# Travel coordinates for NY airports

# JFK airport coordinates, see https://www.travelmath.com/airport/JFK
jfk = (-73.7822222222, 40.6441666667)
ewr = (-74.175, 40.69) # Newark Liberty International Airport, see https://www.travelmath.com/airport/EWR
lgr = (-73.87, 40.77) # LaGuardia Airport, see https://www.travelmath.com/airport/LGA

def get_closeby_data(df, coord, tolerance=1.5, isFromCoord=True):
    if isFromCoord:
        l1='pickup'
        l2='dropoff'
    else:
        l1='dropoff'
        l2='pickup'
    df_new = df.filter(regex=l1).copy()
    df_new['{0}_longitude'.format(l2)] = coord[0]
    df_new['{0}_latitude'.format(l2)] = coord[1]
    df_new['distance'] = df_new.apply(haversine_distance, axis=1)
    df_new['closeby'] = (df_new.distance < tolerance).astype(int)
    
    return df_new

# Going from JFK airport
df_fromjfk = get_closeby_data(df_coordinates, jfk, isFromCoord=True)

# Going to JFK airport
df_tojfk = get_closeby_data(df_coordinates, jfk, isFromCoord=False)

# Going from Newark airport
df_fromnewark = get_closeby_data(df_coordinates, ewr, isFromCoord=True)

# Going to Newark airport
df_tonewark = get_closeby_data(df_coordinates, ewr, isFromCoord=False)

# Going from La Guardia airport
df_fromguardia = get_closeby_data(df_coordinates, lgr, isFromCoord=True)

# Going to La Guardia  airport
df_toguardia = get_closeby_data(df_coordinates, lgr, isFromCoord=False)

df['airport_ride'] = df_fromjfk.closeby | \
                    df_tojfk.closeby | \
                    df_fromnewark.closeby | \
                    df_tonewark.closeby | \
                    df_fromguardia.closeby | \
                    df_toguardia.closeby
                    
non_airport_ride = df.airport_ride[df.airport_ride == 0].count()
airport_ride = df.airport_ride[df.airport_ride == 1].count()
print('non_airport_ride = {0}, airport_rides = {1}'.format(non_airport_ride, airport_ride))
# Adapted from https://www.kaggle.com/maheshdadhich/strength-of-visualization-python-visuals-tutorial

train_fr_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')
train_fr_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')
train_fr = pd.concat([train_fr_1, train_fr_2])
train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
train_df = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')
train = pd.merge(train_df, train_fr_new, on = 'id', how = 'left')

osrm_data = train[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','total_distance', 'total_travel_time', 'number_of_steps']]
osrm_data = osrm_data[select_within_boundingbox(osrm_data, BB)]
osrm_data.describe()
from sklearn.neighbors import NearestNeighbors
osrm_journeys = osrm_data.filter(regex='dropoff|pickup')
neigh = NearestNeighbors(metric='manhattan')
neigh.fit(osrm_journeys)

def get_osrm_distance(row):
    position_data = row.filter(regex='dropoff|pickup')
    nearest_neighbors = neigh.kneighbors([position_data], return_distance=False)
    return osrm_data.iloc[nearest_neighbors.flatten()]['total_distance'].mean()

df['osrm_distance'] = df_coordinates.apply(get_osrm_distance, axis=1)

def get_osrm_time(row):
    position_data = row.filter(regex='dropoff|pickup')
    nearest_neighbors = neigh.kneighbors([position_data], return_distance=False)
    return osrm_data.iloc[nearest_neighbors.flatten()]['total_travel_time'].mean()

df['osrm_time'] = df_coordinates.apply(get_osrm_time, axis=1)

def get_osrm_num_steps(row):
    position_data = row.filter(regex='dropoff|pickup')
    nearest_neighbors = neigh.kneighbors([position_data], return_distance=False)
    return osrm_data.iloc[nearest_neighbors.flatten()]['number_of_steps'].mean()

df['osrm_number_of_steps'] = df_coordinates.apply(get_osrm_num_steps, axis=1)

import matplotlib.pyplot as plt
% matplotlib inline
plt.style.use('seaborn-whitegrid')

plt.figure(figsize=(15,10))
plt.scatter(df.distance[df.airport_ride == 1], df.fare_amount[df.airport_ride == 1], c='r', alpha=0.4, s=10, label='airport ride')
plt.scatter(df.distance[df.airport_ride == 0], df.fare_amount[df.airport_ride == 0], c='b', alpha=0.4, s=10, label='normal ride')
plt.title('Haversine distance against fare prediction', {'fontsize': 20})
plt.xlabel('distance/km', fontsize=16)
plt.ylabel('fare/ $USD', fontsize=16)
plt.legend()
plt.figure(figsize=(15,10))
plt.scatter(df.osrm_distance[df.airport_ride == 1], df.fare_amount[df.airport_ride == 1], c='r', alpha=0.4, s=10, label='airport ride')
plt.scatter(df.osrm_distance[df.airport_ride == 0], df.fare_amount[df.airport_ride == 0], c='b', alpha=0.4, s=10, label='normal ride')
plt.title('OSRM Distance estimates against fare prediction', {'fontsize': 20})
plt.xlabel('distance/m', fontsize=16)
plt.ylabel('fare/ $USD', fontsize=16)
plt.legend()
#https://stackoverflow.com/a/19069028
import statsmodels.api as sm

osrm_distance_km = df.osrm_distance.apply(lambda r: r / 1000)
results = sm.OLS(osrm_distance_km,sm.add_constant(df.distance)).fit()


plt.figure(figsize=(15,10))
plt.scatter(df.distance, osrm_distance_km, alpha=0.4, s=10)
plt.plot(df.distance, df.distance*results.params[1] + results.params[0], '-')

plt.title('Comparison of calculated haversine distance against estimated OSRM distance', {'fontsize': 20})
plt.xlabel('Haversine distance/km', fontsize=16)
plt.ylabel('OSRM estimated distance/km', fontsize=16)
results.summary()
plt.figure(figsize=(15,10))
plt.scatter(df.osrm_time[df.airport_ride == 1], df.fare_amount[df.airport_ride == 1], c='r', alpha=0.4, s=10, label='airport ride')
plt.scatter(df.osrm_time[df.airport_ride == 0], df.fare_amount[df.airport_ride == 0], c='b', alpha=0.4, s=10, label='normal ride')
plt.title('OSRM Time estimates against fare prediction', {'fontsize': 20})
plt.xlabel('time/s', fontsize=16)
plt.ylabel('fare/ $USD', fontsize=16)
plt.legend()
plt.figure(figsize=(15,10))
plt.scatter(df.osrm_number_of_steps[df.airport_ride == 1], df.fare_amount[df.airport_ride == 1], c='r', alpha=0.4, s=10, label='airport ride')
plt.scatter(df.osrm_number_of_steps[df.airport_ride == 0], df.fare_amount[df.airport_ride == 0], c='b', alpha=0.4, s=10, label='normal ride')
plt.title('OSRM Step estimates against fare prediction', {'fontsize': 20})
plt.xlabel('Number of steps', fontsize=16)
plt.ylabel('fare/ $USD', fontsize=16)
plt.legend()