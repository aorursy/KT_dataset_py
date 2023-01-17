# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by click ing run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
green_taxi_jan=  pd.read_csv('/kaggle/input/taxi-data/green_tripdata_2018-01.csv')

print('No.of records in green taxi',green_taxi_jan.shape[0])

green_taxi_jan.head()
yellow_taxi_jan = pd.read_csv('/kaggle/input/taxi-data/yellow_tripdata_2018-01.csv')

print('No.of records in Yellow Taxi',yellow_taxi_jan.shape[0])

yellow_taxi_jan.head(6)
yellow_taxi_jan['tpep_pickup_datetime']   = pd.to_datetime(yellow_taxi_jan['tpep_pickup_datetime'])

yellow_taxi_jan['tpep_dropoff_datetime']  = pd.to_datetime(yellow_taxi_jan['tpep_dropoff_datetime'])

yellow_taxi_jan['pickup_hour'] = yellow_taxi_jan['tpep_pickup_datetime'].dt.hour
# 1.Total Trips taken per Month (both yellow and green taxi's)

print("Total number of trips taken in a month by both green and yellow taxi's :",green_taxi_jan.shape[0]+yellow_taxi_jan.shape[0])
# 2.Average Speed taken by Yellow Taxis per Hour of trips.(Only for yellow taxi)

yellow_taxi_jan['actual_trip_time'] = yellow_taxi_jan.tpep_dropoff_datetime - yellow_taxi_jan.tpep_pickup_datetime

y_group = yellow_taxi_jan.groupby('pickup_hour')

for i in range(24):

    yellow_taxi = y_group.get_group(i)

    time = yellow_taxi[yellow_taxi['trip_distance']>0]['actual_trip_time'].sum()

    total_hours = (time.total_seconds()/(3600))

    #print('Time',total_hours) 

    distance = yellow_taxi_jan['trip_distance'].sum()

    #print('Total distance',distance)

    print('{} Miles per hour is Average Speed of the Yellow taxi between {} to {}'.format(round((distance/total_hours)/31,2),i,i+1))
# Making bins based on the tip's received  (only for yellow taxi's)

print(pd.cut(yellow_taxi_jan.tip_amount,bins = 5))

yellow_taxi_jan['tip_bins'] = pd.cut(yellow_taxi_jan.tip_amount,bins = 5,labels=[1,2,3,4,5])
# Making bins based on the tip's received  (for both  yellow and green taxi's combined)

green_yellow_taxi = pd.DataFrame({'tip_amount':pd.concat([green_taxi_jan['tip_amount'],yellow_taxi_jan['tip_amount']])})

print(pd.cut(green_yellow_taxi.tip_amount,bins = 5))

green_yellow_taxi['tip_bins'] = pd.cut(green_yellow_taxi.tip_amount,bins = 5,labels=[1,2,3,4,5])
# 3.Percentage of Total trips falling in 5 different Tip Bins (for only yellow taxi's)

for x in range(1,6):

    print('{}% of total trips are in bin {}'.format((yellow_taxi_jan[yellow_taxi_jan['tip_bins']==x].shape[0]/yellow_taxi_jan.shape[0])*100,x))
# 3.Percentage of Total trips falling in 5 different Tip Bins (for both yellow and green taxi's)

for x in range(1,6):

    print('{}% of total trips are in bin {}'.format((green_yellow_taxi[green_yellow_taxi['tip_bins']==x].shape[0]/green_yellow_taxi.shape[0])*100,x))
# 2.Average Speed taken by green Taxis per Hour of trips.(Green Taxi)

green_taxi_jan['lpep_pickup_datetime']   = pd.to_datetime(green_taxi_jan['lpep_pickup_datetime'])

green_taxi_jan['lpep_dropoff_datetime']  = pd.to_datetime(green_taxi_jan['lpep_dropoff_datetime'])

green_taxi_jan['pickup_hour'] = green_taxi_jan['lpep_pickup_datetime'].dt.hour

green_taxi_jan['actual_trip_time'] = green_taxi_jan.lpep_dropoff_datetime - green_taxi_jan.lpep_pickup_datetime

y_group = green_taxi_jan.groupby('pickup_hour')

for i in range(24):

    green_taxi = y_group.get_group(i)

    time = green_taxi[green_taxi['trip_distance']>0]['actual_trip_time'].sum()

    total_hours = (time.total_seconds()/(3600))

    #print('Time',total_hours) 

    distance = green_taxi_jan['trip_distance'].sum()

    #print('Total distance',distance)

    print('{} Miles per hour is Average Speed of the Yellow taxi between {} to {}'.format(round((distance/total_hours)/31,2),i,i+1))
# 4.Percentage of total taxi trips taking a certain average speed. Mostly average speed are between 5.50 and 39 mph (only for Yellow Taxi)

print('{}% of total trips are travelling with average speed of between 5.5 and 39 mph'.format(round((yellow_taxi_jan[~(yellow_taxi_jan['pickup_hour'].isin([17,18]))].shape[0]/yellow_taxi_jan.shape[0])*100),2))
# 4.Percentage of total taxi trips taking a certain average speed. Mostly average speed are between 5.50 and 39 mph (for both yellow and green taxi's combined)

print('{}% of total trips are travelling with average speed of between 5.5 and 39 mph'.format(round(((green_taxi_jan[~(green_taxi_jan['pickup_hour'].isin([8,9,14,15,16,17,18,19,20]))].shape[0]+yellow_taxi_jan[~(yellow_taxi_jan['pickup_hour'].isin([17,18]))].shape[0])/(green_taxi_jan.shape[0]+yellow_taxi_jan.shape[0]))*100),2))
# 5.Percentage of total taxi trips travelling a certain average distance. Most trips are within short distances (Only for yellow taxi)

print('Percentage of trips whose average speed is 1.55 miles ',yellow_taxi_jan[yellow_taxi_jan.trip_distance<yellow_taxi_jan.trip_distance.median()].shape[0]/yellow_taxi_jan.shape[0])
# 5.Percentage of total taxi trips travelling a certain average distance. Most trips are within short distances (for both green and yellow taxi's)

green_yellow_taxi['trip_distance'] = pd.concat([green_taxi_jan.trip_distance,yellow_taxi_jan.trip_distance])

print('Percentage of trips whose average speed is 1.55 miles ',green_yellow_taxi[green_yellow_taxi.trip_distance<green_yellow_taxi.trip_distance.median()].shape[0]/green_yellow_taxi.shape[0])