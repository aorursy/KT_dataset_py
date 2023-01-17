%matplotlib inline



import numpy as np 

import pandas as pd

from subprocess import check_output

from tqdm import tqdm

import matplotlib.pyplot as plt

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv', parse_dates=['pickup_datetime', 'dropoff_datetime'])

df.head()
df = df.sample(round(df.shape[0]*0.10), random_state=3435).sort_index()

df.shape
# ref: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

import geopy.distance



distance_feature = np.zeros(df.shape[0])

for k in range(df.shape[0]):

    distance_feature[k] = geopy.distance.vincenty(tuple(df[['pickup_latitude', 'pickup_longitude']].iloc[k,:]), 

                                               tuple(df[['dropoff_latitude', 'dropoff_longitude']].iloc[k,:])).km
# Distance in km

df['travel_distance'] = distance_feature

df['travel_distance'].head()
# The day of the week with Monday=0, Sunday=6

df['day_of_week'] = df['pickup_datetime'].apply(lambda x: x.dayofweek)

df['day_of_week'].head()
df['pct_of_day'] = df['pickup_datetime'].apply(lambda x: (x.hour*60 + x.minute)/1440)

df['pct_of_day'].head()