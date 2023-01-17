#imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline 

from datetime import datetime
# loading just 100k datapoints for illustration
df1 = pd.read_csv('../input/train.csv', nrows = 100_000, parse_dates = ['pickup_datetime']).drop(columns = 'key')
# fairly standard cleanup
def clean(df):
    # dropping rides outside test area
    df=df[(-74.27 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72.95)]
    df=df[(-74.27 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72.95)]
    df=df[(40.56 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 41.71)]
    df=df[(40.56 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 41.71)]
    # Remove passenger count outliers
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 6)]
    # Remove possible fare outliers
    df=df[(2.5 <= df['fare_amount']) & (df['fare_amount'] <= 250)]

    return df
# distances 
def distance (x1,y1,x2,y2):# x=longitude, y=latitude, 1=start, 2=finish
    return  np.sqrt(((x2-x1)* 50)** 2 + ((y2-y1)*69.172)**2)

# EXCERCISE IN DISTANCE ALONG MANHATTAN GRID ALIGNED AT 29 DEGREE ANGLE
# Manhattan grid is tilted at 29 degrees angle (https://trefethen.net/2013/09/29/manhattan-is-tilted-at-a-29-degree-angle/)
# sin (29) ~ 0.4848096, cos (29) ~ 0,8746197
# 1 degree of latitude  = 69.172 miles, 1 degree of longitude = 50 miles
def distance29(x1,y1,x2,y2):# x=longitude, y=latitude, 1=start, 2=finish
    dist_avenue = (x2-x1) * 0.4848096 * 50 + (y2-y1) * 0.8746197 * 69.172
    dist_street = (x2-x1) * 0.8746197 * 50 - (y2-y1) * 0.4848096 * 69.172
    return np.abs(dist_avenue) + np.abs(dist_street)
# It has about same correlation to fare_amount but is helpful in improving predictions

def add_location_factors(df):
    df['distance'] = distance(df['pickup_latitude'], df['pickup_longitude'],df['dropoff_latitude'], df['dropoff_longitude'])
    #df['distance29'] = np.vectorize(distance29)(df['pickup_latitude'], df['pickup_longitude'],df['dropoff_latitude'], df['dropoff_longitude'])
    #df['distance29'] = distance29(df['pickup_latitude'], df['pickup_longitude'],df['dropoff_latitude'], df['dropoff_longitude'])
    #df['zero'] = df['distance'].apply(lambda x: x < 0.01 )
    return (df)
import matplotlib.path as mpltPath
manhattan_path = mpltPath.Path([[40.698,-74.019 ],[40.757,-74.014 ],[ 40.881745, -73.934875 ],[ 40.872186, -73.909654 ],\
                     [40.834051, -73.934120],[ 40.809238, -73.933307 ],[40.798337, -73.927591],[ 40.773668, -73.941674 ],\
                    [40.741346, -73.966607],[40.707832, -73.974694]])

def is_within(x,y,path):  #(latitude, longitude)
    return path.contains_point([x,y])
# inspired by https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python 

def manhattanize (df):
    df ['manh_pu'] = np.vectorize(is_within)(df['pickup_latitude'],df['pickup_longitude'],manhattan_path)
    df ['manh_do'] = np.vectorize(is_within)(df['dropoff_latitude'],df['dropoff_longitude'],manhattan_path)
    df ['manh'] = df ['manh_pu'] & df ['manh_do'] 
    return df
df1 = clean(df1)
df1 = add_location_factors(df1)
df1 = manhattanize(df1)
print ('{:.3}% of the sample rides are wholly within Manhattan'.format(df1[df1.manh].shape[0] / df1.shape[0] * 100))
df1.head(40)
# big majority of the rides are within Manhattan.
# this function will also be used with the test set below
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & \
           (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) & \
           (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) & \
           (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])
            
# load extra image to zoom in on NYC
BB_zoom = (-74.3, -73.7, 40.5, 40.9)
nyc_map_zoom = plt.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')
# this function will be used more often to plot data on the NYC map
pu_w = df1[df1['manh_pu'] == True].sample(1000)
pu_o = df1[df1['manh_pu'] == False].sample(2000)
s=10
alpha=0.2
BB = BB_zoom
fig, axs = plt.subplots(1, 2, figsize=(16,10))
axs[0].scatter(pu_w.pickup_longitude, pu_w.pickup_latitude, zorder=1, alpha=alpha, c='r', s=s)
axs[0].scatter(pu_o.pickup_longitude, pu_o.pickup_latitude, zorder=1, alpha=alpha, c='b', s=s)
axs[0].set_xlim((BB[0], BB[1]))
axs[0].set_ylim((BB[2], BB[3]))
axs[0].set_title('Pickup locations inside / outside Manhattan')
axs[0].imshow(nyc_map_zoom, zorder=0, extent=BB)

do_w = df1[df1['manh_do'] == True].sample(1000)
do_o = df1[df1['manh_do'] == False].sample(2000)

axs[1].scatter(do_w.dropoff_longitude, do_w.dropoff_latitude, zorder=1, alpha=alpha, c='r', s=s)
axs[1].scatter(do_o.dropoff_longitude, do_o.dropoff_latitude, zorder=1, alpha=alpha, c='g', s=s)
axs[1].set_xlim((BB[0], BB[1]))
axs[1].set_ylim((BB[2], BB[3]))
axs[1].set_title('Dropoff locations inside / outside Manhattan')
axs[1].imshow(nyc_map_zoom, zorder=0, extent=BB)