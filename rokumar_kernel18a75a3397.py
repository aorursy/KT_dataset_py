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
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib as mp
import seaborn as sb
from datetime import datetime as dt
import calendar as cd

long_to_miles = 55.2428
lat_to_miles = 69.0

def prepare_clean_data_set():
    #count = 1
    chunksize = 10 ** 6
    rows =  10 ** 6
    skiprowcount = 0
    filename = "../input/train.csv"
    
    #for chunk in pd.read_csv(filename):
    chunk = pd.read_csv(filename, nrows=10)
    passenger_cond = (chunk['passenger_count'] > 0) & (chunk['passenger_count'] < 5)
    fare_cond = (chunk['fare_amount'] > 3) & (chunk['fare_amount'] < 500)
    state_long_cond = (chunk['pickup_longitude'] <= -71.7517) & \
                      (chunk['pickup_longitude'] >= -79.76) | \
                      (chunk['dropoff_longitude'] <= -71.7517) & \
                      (chunk['dropoff_longitude'] >= -79.76)
    state_lat_cond =  (chunk['pickup_latitude'] >= 40.4772) & \
                      (chunk['pickup_latitude'] <= 45.0153) | \
                      (chunk['dropoff_latitude'] >= 40.4772) & \
                      (chunk['dropoff_latitude'] <= 45.0153)
    nyc_long_cond =   (chunk['pickup_longitude'] >= -74.04) & \
                      (chunk['pickup_longitude'] <= -73.70) | \
                      (chunk['dropoff_longitude'] >= -74.04) & \
                      (chunk['dropoff_longitude'] <= -73.70)
    nyc_lat_cond =    (chunk['pickup_latitude'] <= 40.91) & \
                      (chunk['pickup_latitude'] >= 40.54) | \
                      (chunk['dropoff_latitude'] >= 40.54) & \
                      (chunk['dropoff_latitude'] <= 40.91)

    chunk = chunk.dropna()
    chunk = chunk[fare_cond]
    chunk = chunk[state_long_cond]
    chunk = chunk[state_lat_cond]
    chunk = chunk[passenger_cond]
    chunk = chunk[nyc_lat_cond]
    chunk = chunk[nyc_long_cond]

    chunk['long_mile_diff']  = (((chunk.pickup_longitude - chunk.dropoff_longitude).abs()) * long_to_miles)
    chunk['lat_mile_diff']   = (((chunk.pickup_latitude - chunk.dropoff_latitude).abs()) * lat_to_miles)
    chunk['euclid_dist']     = np.sqrt(chunk['long_mile_diff'] ** 2 + chunk['long_mile_diff'] ** 2)
    chunk['epoch']           = chunk['pickup_datetime'].\
                                 apply(lambda val: dt.strptime(val, "%Y-%m-%d %H:%M:%S UTC").toordinal())
    chunk['pickup_datetime'] = pd.to_datetime(chunk['pickup_datetime'], format='%Y-%m-%d %H:%M:%S UTC')
    chunk['day']     = chunk['pickup_datetime'].apply(lambda d: d.day)
    chunk['month']   = chunk['pickup_datetime'].apply(lambda m : m.month)
    chunk['year']    = chunk['pickup_datetime'].apply(lambda y: y.year)
    chunk['hour']    = chunk['pickup_datetime'].apply(lambda hr: hr.hour)
    chunk['weekday'] = chunk['pickup_datetime'].\
                        apply(lambda wd:  1 if (wd.weekday() == 5 or wd.weekday() == 6) else 2)
    chunk['farepermile'] = chunk.apply(lambda r: round(r['euclid_dist']/r['fare_amount'], 2), axis = 1)

    farepermilecond = (chunk['farepermile'] >=  (0.139048 - (2 * 0.444676))) & \
          (chunk['farepermile'] <= (0.139048 + (2 * 0.444676)))

    euclid_dist_cond = (chunk['euclid_dist'] > 0) & (chunk['euclid_dist'] < 50) 

    chunk = chunk[euclid_dist_cond]
    chunk = chunk[farepermilecond]
    chunk.to_csv("output.csv", header=True)
    
#     with open('my_csv.csv', 'a') as f:
#         if(count == 1):
#             chunk.to_csv(f, header=True)
#         else: 
#             chunk.to_csv(f, header=False)
#         chunk.to_csv("cleaned_data.csv")

    #count += 1

prepare_clean_data_set()

pwd
