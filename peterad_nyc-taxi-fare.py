import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import matplotlib
from sklearn.cluster import MiniBatchKMeans, KMeans
#% matplotlib inline
import os
import time
import calendar

print(os.listdir("../input"))
print(os.listdir("../input/nyc-taxi-fare-osrm"))
train_df =  pd.read_csv('../input/nyc-taxi-fare-osrm/train_osrm_50M.csv')
#excludes rows outside of bounding box latitudes and longitudes
#from https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration

#long_min, long_max, lat_min, lat_max = (-74.5, -72.8, 40.5, 41.8) #loose constraints
#map_img = mpimg.imread('https://aiblog.nl/download/nyc_-74.5_-72.8_40.5_41.8.png')
long_min, long_max, lat_min, lat_max = (-74.3, -73.7, 40.5, 40.9) #tight constraints
#map_img = mpimg.imread('https://aiblog.nl/download/nyc_-74.3_-73.7_40.5_40.9.png')
#from https://imgur.com/a/nMkieh6
map_img = mpimg.imread('https://i.imgur.com/hXaTTqp.png')

def bounding_box(df):
    return df[(df.pickup_longitude >= long_min) & (df.pickup_longitude <= long_max) & \
           (df.pickup_latitude >= lat_min) & (df.pickup_latitude <= lat_max) & \
           (df.dropoff_longitude >= long_min) & (df.dropoff_longitude <= long_max) & \
           (df.dropoff_latitude >= lat_min) & (df.dropoff_latitude <= lat_max)]

min_distance, max_distance, min_duration, max_duration = (100.,60000.,30.,4000.)
def distance_duration_box(df):
    return df[(df.distance >= min_distance) & (df.distance <= max_distance) & \
           (df.duration >= min_duration) & (df.duration <= max_duration)]

min_fare, max_fare = (3.,200.)
def fare_box(df):
    return df[(df.fare_amount >= min_fare) & (df.fare_amount <= max_fare)]

#runs all data cleaning methods
def clean_data(df):
    df = bounding_box(df)
    df = distance_duration_box(df)
    return df

"""
#This is what a function that uses OSRM to calculate distance looks like interacting with a local server.
#See here to install the backend: https://github.com/Project-OSRM/osrm-backend/wiki
#Here to install frontend: https://pypi.org/project/osrm-py/

import osrm
client = osrm.Client(host='http://localhost:5000')
def osrm_calc(long_in,lat_in,long_out,lat_out):
    coordinates = [[long_in,lat_in],[long_out,lat_out]]
    response = client.route(coordinates=coords_nest)
    return response['routes'][0]['distance'],response['routes'][0]['duration']
"""

#bins the latitude and longitude variables to make it possible to create heatmaps
def latitude_longitude_binning(df):
    df['dropoff_longitude_bin'] = pd.cut(df.dropoff_longitude, bins=50)
    df['dropoff_latitude_bin'] = pd.cut(df.dropoff_latitude, bins=50)
    df['pickup_longitude_bin'] = pd.cut(df.pickup_longitude, bins=50)
    df['pickup_latitude_bin'] = pd.cut(df.pickup_latitude, bins=50)
    return df

def distance_duration_binning(df):
    df['distance_bin'] = pd.cut(df.distance, bins=50)
    df['duration_bin'] = pd.cut(df.duration, bins=50)
    return df

fare_bins = [3.,5.,7.,9.,11.,13.,15.,17.,19.,21.,23.,25.,27.,29.,31.,33.,35.,37.,39.,41.,43.,45.,47.,49.,51.,53.,55.,60.,65.,70.,75.,80.,90.,100.,125.,150.,200.]
def fare_binning(df):
    df['fare_amount_bin'] = pd.cut(df.fare_amount, bins = fare_bins)
    return df

def apply_clusters(df):
    df['pickup_cluster'] = clusters.predict(df[['pickup_longitude','pickup_latitude']])
    df['dropoff_cluster'] = clusters.predict(train_df[['dropoff_longitude','dropoff_latitude']])
    return df

#largely from https://www.kaggle.com/aiswaryaramachandran/eda-and-feature-engineering
def time_columns(df):
    df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'],format='%Y-%m-%d %H:%M:%S UTC')
    df['pickup_date']= df['pickup_datetime'].dt.date
    df['pickup_day']=df['pickup_datetime'].apply(lambda x:x.day)
    df['pickup_hour']=df['pickup_datetime'].apply(lambda x:x.hour)
    df['pickup_day_of_week']=df['pickup_datetime'].apply(lambda x:x.weekday())
    df['pickup_month']=df['pickup_datetime'].apply(lambda x:x.month)
    df['pickup_year']=df['pickup_datetime'].apply(lambda x:x.year)
    return df

#runs all data creating methods (do not add training only columns)
def create_columns(df):
    df = latitude_longitude_binning(df)
    df = apply_clusters(df)
    df = distance_duration_binning(df)
    df = time_columns(df)
    return df

def heatmap_on_pic(pv,vmax=None, cmap=matplotlib.cm.YlGn):
    fig, ax = plt.subplots(figsize=(18,14))
    #optional kwargs
    kwargs = {}
    if vmax is not None: kwargs['vmax'] = vmax
    kwargs['cmap'] = cmap
    
    ax = sns.heatmap(pv, ax=ax, alpha = 0.8, zorder = 2, **kwargs)
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    _ = ax.imshow(map_img,
                   aspect = ax.get_aspect(),
                   extent = ax.get_xlim() + ax.get_ylim(),
                   zorder = 1)
    return fig,ax
fig1, ax1 = plt.subplots()
train_df.pickup_longitude.hist(ax=ax1, bins=100, bottom=0.1, figsize=(14,3))
ax1.set_yscale('log')
_ = ax1.set_xlabel('longitude')

fig2, ax2 = plt.subplots()
train_df.pickup_latitude.hist(ax=ax2, bins=100, bottom=0.1, figsize=(14,3))
ax2.set_yscale('log')
_ = ax2.set_xlabel('latitude')
train_df = bounding_box(train_df)
train_df = latitude_longitude_binning(train_df)
train_df.head()
fig1, ax1 = plt.subplots()
train_df.pickup_longitude.hist(ax=ax1, bins=100, bottom=0.1, figsize=(14,3))
#ax1.set_yscale('log')
_ = ax1.set_xlabel('longitude')

fig2, ax2 = plt.subplots()
train_df.pickup_latitude.hist(ax=ax2, bins=100, bottom=0.1, figsize=(14,3))
#ax2.set_yscale('log')
_ = ax2.set_xlabel('latitude')
fig1, ax1 = plt.subplots()
train_df.distance.hist(ax=ax1, bins=100, bottom=0.1, figsize=(14,3))
ax1.set_xlim(0.,60000.)
ax1.set_yscale('log')
_ = ax1.set_xlabel('distance (meters)')

fig2, ax2 = plt.subplots()
train_df.duration.hist(ax=ax2, bins=100, bottom=0.1, figsize=(14,3))
ax2.set_xlim(0.,4000.)
ax2.set_yscale('log')
_ = ax2.set_xlabel('duration (seconds)')

fig3, ax3 = plt.subplots()
train_df.fare_amount.hist(ax=ax3, bins=100, bottom=0.1, figsize=(14,3))
ax3.set_xlim(-10.,200.)
ax3.set_yscale('log')
_ = ax3.set_xlabel('fare (dollars)')
train_df = distance_duration_box(train_df)
train_df = fare_box(train_df)
train_df = distance_duration_binning(train_df)
train_df = fare_binning(train_df)
train_df.head()
pv1 = pd.pivot_table(train_df,aggfunc='size',columns='pickup_longitude_bin',index='pickup_latitude_bin',fill_value=0.0,dropna=False)
fig1,ax1 = heatmap_on_pic(pv1,vmax=5000.,cmap=matplotlib.cm.Reds)
plt.title('Pickup',fontsize=20)
plt.show()
pv2 = pd.pivot_table(train_df,aggfunc='size',columns='dropoff_longitude_bin',index='dropoff_latitude_bin',fill_value=0.0,dropna=False)
fig2,ax2 = heatmap_on_pic(pv2,vmax=5000.,cmap=matplotlib.cm.Blues)
plt.title('Dropoff',fontsize=20)
plt.show()
clusters = KMeans(n_clusters=15, random_state=0).fit(train_df[:100000][['pickup_longitude','pickup_latitude']])
train_df = apply_clusters(train_df)
train_df.head()
h = .0005
xx, yy = np.meshgrid(np.arange(long_min,long_max,h),np.arange(lat_min,lat_max,h))
Z = clusters.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(figsize=(18,14))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=ListedColormap(sns.diverging_palette(220, 70, n=15).as_hex()),#plt.cm.tab20b,
           aspect='auto', origin='lower', alpha=0.7, zorder=2)
centroids = clusters.cluster_centers_
labels = clusters.predict(centroids)
for label, x, y in zip(labels,centroids[:,0],centroids[:,1]):
    plt.annotate(label,xy=(x,y),fontsize='15')
plt.ylim(lat_min,lat_max)
plt.xlim(long_min,long_max)
_ = ax.imshow(map_img,
          aspect = ax.get_aspect(),
          extent = ax.get_xlim() + ax.get_ylim(),
          zorder = 1)
plt.show()
for i,district in zip([2,3,4,12],['JFK','Lower Manhatten','LaGuardia','Upper Manhatten']):
    pv = pd.pivot_table(train_df[(train_df.pickup_cluster==i)],values='fare_amount',columns='dropoff_longitude_bin',index='dropoff_latitude_bin',fill_value=0.0,dropna=False)
    fig,ax = heatmap_on_pic(pv,vmax=70.)
    plt.title("Pickup from {0} (cluster {1})".format(district,i), fontsize=20)
    plt.show()
    pv = pd.pivot_table(train_df[(train_df.distance>1000.) & (train_df.distance<20000.)],columns='distance_bin',index='duration_bin',values='fare_amount')
    fig, ax = plt.subplots(figsize=(18,14))
    ax = sns.heatmap(pv, ax=ax, vmax=40, cmap=matplotlib.cm.Wistia)
    ax.invert_yaxis()
    plt.show()
    pv = pd.pivot_table(train_df,aggfunc='size',columns='distance_bin',index='fare_amount_bin',fill_value=0.0)
    fig, ax = plt.subplots(figsize=(18,14))
    ax = sns.heatmap(pv, ax=ax, vmax=20000., cmap=matplotlib.cm.YlOrRd)
    ax.invert_yaxis()
    plt.show()
    pv = pd.pivot_table(train_df[(train_df.pickup_cluster==2) & (train_df.dropoff_cluster==3)],aggfunc='size',columns='distance_bin',index='fare_amount_bin',fill_value=0.0)
    fig, ax = plt.subplots(figsize=(18,14))
    ax = sns.heatmap(pv, ax=ax, alpha = 0.8, vmax= 200, cmap=matplotlib.cm.Blues)
    ax.invert_yaxis()
    plt.show()
train_df = time_columns(train_df)
train_df.head()
train_df.describe()
fig1, ax1 = plt.subplots()
bins = range(2009,2017)
train_df.pickup_year.hist(ax=ax1, bins=bins, bottom=0.1, figsize=(14,3), align='left')
_ = ax1.set_xlabel('Year (AD)')

fig2, ax2 = plt.subplots()
bins = range(8)
train_df.pickup_day_of_week.hist(ax=ax2, bins=bins, bottom=0.1, figsize=(14,3), align='left')
_ = ax2.set_xlabel('Days since Midnight on Sunday')

fig3, ax3 = plt.subplots()
bins = range(0,25)
train_df.pickup_hour.hist(ax=ax3, bins=bins, bottom=0.1, figsize=(14,3),align='left')
_ = ax3.set_xlabel('Hour since Midnight')
fig, ax = plt.subplots(figsize=(14,10))
ax = sns.boxplot(data = train_df[(train_df.pickup_cluster==2) & (train_df.dropoff_cluster==3) & (train_df.pickup_hour>5)], x='pickup_year',y='fare_amount', ax=ax, showfliers=False)
fig1, ax1 = plt.subplots()
train_df[(train_df.pickup_cluster==12) & (train_df.dropoff_cluster==3) & (train_df.pickup_year<2013)].distance.hist(ax=ax1, bins=20, figsize=(14,3))
_ = ax1.set_xlabel('Distance (meters)')

fig2, ax2 = plt.subplots()
train_df[(train_df.pickup_cluster==12) & (train_df.dropoff_cluster==3) & (train_df.pickup_year<2013) & (train_df.distance >7000.) & (train_df.distance < 9000.)].duration.hist(ax=ax2, bins=20, figsize=(14,3))
_ = ax.set_xlabel('Durations (seconds)')
t = plt.title('7km < distance < 9km', fontsize=12)
fig, ax = plt.subplots(figsize=(14,10))
ax = sns.boxplot(
    data = train_df[(train_df.pickup_cluster==12) & (train_df.dropoff_cluster==3) & (train_df.duration >450.) & (train_df.duration < 650.) & (train_df.distance >6000.) & (train_df.distance < 9000.)], 
    x='pickup_year',y='fare_amount', ax=ax, showfliers=False)
fig, ax = plt.subplots(figsize=(14,10))
ax = sns.boxplot(
    data = train_df[(train_df.pickup_cluster==12) & (train_df.dropoff_cluster==3) & (train_df.pickup_year<2013) & (train_df.duration >450.) & (train_df.duration < 650.) & (train_df.distance >6000.) & (train_df.distance < 9000.)], 
    x='pickup_hour',y='fare_amount', ax=ax, showfliers=False)
fig, ax = plt.subplots(figsize=(14,10))
ax = sns.boxplot(
    data = train_df[(train_df.pickup_cluster==12) & (train_df.dropoff_cluster==3) & (train_df.pickup_year<2013) & (train_df.duration >450.) & (train_df.duration < 650.) & (train_df.distance >6000.) & (train_df.distance < 9000.) & (train_df.pickup_hour>8) & (train_df.pickup_hour<20)], 
    x='pickup_day_of_week',y='fare_amount', ax=ax, showfliers=False)
fig, ax = plt.subplots(figsize=(14,10))
ax = sns.boxplot(data = train_df[(train_df.pickup_cluster==12) & (train_df.dropoff_cluster==3)& (train_df.pickup_year<2013) & (train_df.duration >450.) & (train_df.duration < 650.) & (train_df.distance >6000.) & (train_df.distance < 9000.)], x='passenger_count',y='fare_amount', ax=ax, showfliers=False)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from math import sqrt

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
drop = ['key','passenger_count','dropoff_longitude_bin', 'dropoff_latitude_bin', 'pickup_longitude_bin', 'pickup_latitude_bin', 'distance_bin', 'duration_bin', 'fare_amount_bin', 'pickup_date', 'pickup_datetime']
y = train_df['fare_amount']
X = train_df.drop(columns=['fare_amount'])
X = X.drop(columns=drop)
X = pd.get_dummies(X, columns=['pickup_cluster','dropoff_cluster'])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
train_X.head()
params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
    }
train_lgb = lgb.Dataset(train_X, train_y)
trained_lgb = lgb.train(params, train_set = train_lgb, num_boost_round=300)
predicted_y = trained_lgb.predict(test_X, num_iteration = trained_lgb.best_iteration)
print('LGBM RMSE: {0}'.format(sqrt(mean_squared_error(test_y,predicted_y))))
def baseline_model():
    model = Sequential()
    model.add(Dense(12, input_dim=41, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
"""
seed = 7
np.random.seed(seed)
trained_snn = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
with tf.device('/gpu:0'):
    trained_snn.fit(train_X.values,train_y.values, epochs=100, batch_size=5, verbose =2)
predicted_y = trained_snn.predict(test_X)
print('SNN RMSE: {0}'.format(sqrt(mean_squared_error(test_y,predicted_y))))
"""