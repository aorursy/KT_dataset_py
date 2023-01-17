import pandas as pd
import numpy as np
import geopy.distance
import dask.dataframe as dd
from dask.multiprocessing import get
! ls ../input
%%time
data = pd.read_csv('../input/train.csv', nrows=int(2e7))
print(data.shape)
data.dropna(inplace=True)
print(data.shape)
# drop data with lat/lon outside of newyork range
# range are calculated through this link : https://www.mapdevelopers.com/geocode_bounding_box.php

ny_lat_min =  40.477399
ny_lat_max = 40.917577
ny_lon_max = -73.700272
ny_lon_min =  -74.259090
#--------------------------------------------
indices_to_drop = data[(data.pickup_latitude < ny_lat_min) | (data.pickup_latitude > ny_lat_max) |
                      (data.dropoff_latitude < ny_lat_min) | (data.dropoff_latitude > ny_lat_max)|
                      (data.pickup_longitude < ny_lon_min) | (data.pickup_longitude > ny_lon_max) |
                      (data.dropoff_longitude < ny_lon_min) | (data.dropoff_longitude > ny_lon_max)].index
print (len(indices_to_drop))
#----------------------------------------------
data.drop(index=indices_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.shape)
%%time
# Compute Distance using geopy.. [6 cores/ 6 partitions and 20M rows takes ~ 25 minutes]
cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
ddata = dd.from_pandas(data[cols], npartitions=4)

def compute_distance(lat1, lon1, lat2, lon2):
    return round(geopy.distance.distance((lat1,lon1), (lat2, lon2)).km,2)

data['distance'] = ddata.map_partitions(lambda df: df.apply((lambda row: compute_distance(*row)), axis=1))\
    .compute(get=get)
data.head(3)
%%time
#data.to_hdf('./data/v2/sample_dist_2M.hdf', key='sample_dist_2M')
# for a normal taxi the maximum number of passenger would be 4 or 5,
# maybe for a larger cap with two large backseats it would be 7. 
# I'll drop any columns with passenger larger than 7 or lower than 1 ( so obvious.)

indices_to_drop = data[(data.passenger_count < 1) | (data.passenger_count > 7)].index
print (len(indices_to_drop))
#-----------------------------------
data.drop(index=indices_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.shape)
# drop outliers also
Q1 = data.passenger_count.quantile(q=.25)
Q3 = data.passenger_count.quantile(q=.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print('Q1 = {}, Q3 = {}, IQR = {},\nlower limit = {}, upper limit = {}'\
     .format(Q1,Q3,IQR,lower,upper))
#--------------------
# But instead i'll drop > .05% and < .95% so I keep as many rows as possible.
lower = data.passenger_count.quantile(.05)
upper = data.passenger_count.quantile(.95)
print('.05 quantile = {}, .95 quantile = {}'.format(lower, upper))
#-----------
indices_to_drop = data[(data.passenger_count < lower)|(data.passenger_count > upper)].index
print('number of indices to drop : ', len(indices_to_drop))
#------------
data.drop(index=indices_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.shape)
# first delete all negative/zero fare_amount as they won't be real
indices_to_drop = data[data.fare_amount <= 0].index
print (len(indices_to_drop))
#---------------------
data.drop(index=indices_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.shape)
# drop fare_amount outliers also
Q1 = data.fare_amount.quantile(q=.25)
Q3 = data.fare_amount.quantile(q=.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print('Q1 = {}, Q3 = {}, IQR = {},\nlower limit = {}, upper limit = {}'\
     .format(Q1,Q3,IQR,lower,upper))
#----------------------
# and here also i'll drop .05 and .95
lower = data.fare_amount.quantile(.05)
upper = data.fare_amount.quantile(.95)
print('.05 quantile = {}, .95 quantile = {}'.format(lower, upper))
#--------------
indices_to_drop = data[(data.fare_amount < lower)|(data.fare_amount > upper)].index
print('number of rows to drop : ', len(indices_to_drop))
#--------------------------
data.drop(index=indices_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)
print(data.shape)
%%time

year = data.pickup_datetime.apply(lambda d: d[:4])
month = data.pickup_datetime.apply(lambda d: d[5:7])
hour = data.pickup_datetime.apply(lambda d: d[11:13])

year = year.astype(int)
month = month.astype(int)
hour = hour.astype(int)

data['year'] = year
data['month'] = month
data['hour'] = hour
data.head(3)
%%time
#data.to_hdf('./data/v2/sample_dist_2M_ready.hdf', key='sample_dist_2M_ready')
#or
#data = pd.read_hdf('./data/v2/sample_dist_2M_ready.hdf')
data.corr()
cols_to_keep = ['fare_amount', 'pickup_longitude', 'dropoff_longitude', 'distance', 'year']

data = data[cols_to_keep]

data.rename(columns={'fare_amount':'target',
            'pickup_longitude':'pickup',
            'dropoff_longitude':'dropoff'}, inplace=True)
print(data.shape)
data.head(3)
from sklearn.preprocessing import MinMaxScaler
%%time
scaler = MinMaxScaler()
scaled_data = data.copy()
cols = ['pickup', 'dropoff', 'distance', 'year']
scaled_data[cols] = scaler.fit_transform(data[cols])

#scaled_data.target = np.log1p(data.target) # remember to convert the results using np.expm1()
scaled_data.head(3)
%%time
#scaled_data.to_hdf('./data/v2/ready_scaled_20M.hdf', 'ready_scaled_20M')
scaled_data.corr()
