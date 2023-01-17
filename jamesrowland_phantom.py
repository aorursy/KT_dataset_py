"""

The first step is to import the data.

I mostly use the numpy/pandas/... stack. 

I am fluent in TensorFlow due to my research. 

I am comfortable learning whatever stack is prefered at your company. 

For example, I found the package mplleafleat and used it for the first time in this jupyter notebook.

"""



import mplleaflet

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.neighbors import RadiusNeighborsRegressor



phantom_pos = [[-122.094051], [37.4176655]]

tower_pos = [[-122.0911, -122.092], [37.4193, 37.4203]]
df_signal = pd.read_csv('../input/ds_homework_signal.csv', parse_dates=['time'])

df_gps = pd.read_csv('../input/ds_homework_gps.csv', parse_dates=['time'])

df = pd.merge(

    df_gps, df_signal,

    on=['connection_id','time'], how='inner'

).drop_duplicates(['connection_id','time'])

df.head()
print("GPS coordinates")

plt.figure(figsize=(8,8))

skip = 100

plt.scatter(df['lng'].iloc[::skip], df['lat'].iloc[::skip], c='r', s=100, alpha=1) 

plt.scatter(*phantom_pos, c='k', s=800)

plt.scatter(*phantom_pos, c='w', s=250)

mplleaflet.display(tiles='esri_aerial')
from math import sin, cos, sqrt, atan2, radians



# First sort by connection_id and timestamp:

df = df.sort_values(by=['connection_id', 'time'])



# make a copy for temporary dataframe manipulation

df_copy = df.copy()



def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):

    R = 3959 # Radius of the earth in miles

    dLat = radians(lat2-lat1)

    dLon = radians(lon2-lon1)

    rLat1 = radians(lat1)

    rLat2 = radians(lat2)

    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2) 

    c = 2 * atan2(sqrt(a), sqrt(1-a))

    d = R * c # Distance in km

    return d



def calc_velocity(dr, dt):

    """Return 0 if time_start == time_end, avoid dividing by 0"""

    if dt>0:

        # speed in miles per hour

        velocity = 60*60*dr/dt

    else:

        velocity = 0

    return(velocity)



# Group the sorted dataframe by connection_id, then get the coordinate for timestep t+dt

df_copy['lat_shift'] = df_copy.groupby('connection_id').apply(lambda x: x['lat'].shift().fillna(x['lat'])).values

df_copy['lng_shift'] = df_copy.groupby('connection_id').apply(lambda x: x['lng'].shift().fillna(x['lng'])).values

df_copy['time_shift'] = df_copy.groupby('connection_id').apply(lambda x: x['time'].shift().fillna(x['time'])).values



# create a new column for distance

df_copy['dr'] = df_copy.apply(

    lambda row: getDistanceFromLatLonInKm(

        lat1=row['lat_shift'],

        lon1=row['lng_shift'],

        lat2=row['lat'],

        lon2=row['lng']

    ),

    axis=1

)



# create a new column for time delta

df_copy['dt'] = (df_copy['time'] - df_copy['time_shift']).map(lambda x: x.total_seconds())



# create a new column for speed

df['speed'] = df_copy.apply(

    lambda row: calc_velocity(

        dr=row['dr'],

        dt=row['dt']

    ),

    axis=1

)

df['log(speed)'] = np.log1p(df['speed'])
print("Signal-speed correlations")

fig, axes = plt.subplots(2,2,figsize=(14,12))

df.plot.scatter('log(speed)','rsrp',ax=axes[0,0])

df.plot.scatter('log(speed)','sinr',ax=axes[1,0])

df.plot.scatter('rsrp','sinr',ax=axes[0,1]);

sns.heatmap(df.drop(['connection_id','speed'],axis=1).corr().abs(), ax=axes[1,1]);
print("Entries with speed>80mph")

df.loc[df['speed']>80]
print("Trip with speed>80mph")

plt.figure(figsize=(8,8))

df_temp = df.loc[df['connection_id']==15501789206067040733].iloc[:100].append(df.loc[df['connection_id']==15501789206067040733].iloc[101::20])

plt.scatter(df_temp['lng'], df_temp['lat'], c='k', s=600)

plt.scatter(df_temp['lng'], df_temp['lat'], c=df_temp['speed'], s=300)

plt.scatter(*phantom_pos, c='k', s=800)

plt.scatter(*phantom_pos, c='w', s=250)

#plt.scatter(*df_temp[['lng','lat']].values.transpose(), c=df_temp['speed']);

mplleaflet.display(tiles='esri_aerial')
print("RSRP map")

# fit the rsrp signal to nearest neighbor interpolation

# the fit should help us find the source of the signal

model = RadiusNeighborsRegressor(radius = 1, weights='distance')



model.fit(df[['lng','lat']], df['rsrp'])



plt.figure(figsize=(8,8))

px = 30

py = 30

xmin = df['lng'].min()

xmax = df['lng'].max()

xs = np.linspace(xmin, xmax, px)

ymin = df['lat'].min()

ymax = df['lat'].max()

ys = np.linspace(ymin, ymax, py)

rs = np.reshape(np.array(np.meshgrid(xs, ys)).transpose(),(px*py,2))

zs = model.predict(rs)

plt.scatter(*tower_pos, c='k', s=800)

plt.scatter(*tower_pos, c='r', s=400)

plt.contour(

    *(np.reshape(arr,(py,px)) for arr in (rs[:,0], rs[:,1], zs)),

    levels=10, linewidths=12.0, colors='k'

)

plt.contour(

    *(np.reshape(arr,(py,px)) for arr in (rs[:,0], rs[:,1], zs)),

    levels=10, linewidths=6.0

)

plt.scatter(*phantom_pos, c='k', s=800)

plt.scatter(*phantom_pos, c='w', s=250)

mplleaflet.display(tiles='esri_aerial')
print("SINR map")



# fit the rsrp signal to nearest neighbor interpolation

# the fit should help us find the source of the signal

model = RadiusNeighborsRegressor(radius = 1, weights='distance')



model.fit(df[['lng','lat']], df['sinr'])



plt.figure(figsize=(8,8))

px = 30

py = 30

xmin = df['lng'].min()

xmax = df['lng'].max()

xs = np.linspace(xmin, xmax, px)

ymin = df['lat'].min()

ymax = df['lat'].max()

ys = np.linspace(ymin, ymax, py)

rs = np.reshape(np.array(np.meshgrid(xs, ys)).transpose(),(px*py,2))

zs = model.predict(rs)

plt.scatter(*tower_pos, c='k', s=800)

plt.scatter(*tower_pos, c='r', s=400)

plt.contour(

    *(np.reshape(arr,(py,px)) for arr in (rs[:,0], rs[:,1], zs)),

    levels=10, linewidths=12.0, colors='k'

)

plt.contour(

    *(np.reshape(arr,(py,px)) for arr in (rs[:,0], rs[:,1], zs)),

    levels=10, linewidths=6.0

)

plt.scatter(*phantom_pos, c='k', s=800)

plt.scatter(*phantom_pos, c='w', s=250)

mplleaflet.display(tiles='esri_aerial')