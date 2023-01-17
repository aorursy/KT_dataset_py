# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import animation
%matplotlib inline
import seaborn as sns
plt.style.use('fivethirtyeight')


from bokeh.io import output_notebook,show
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.palettes import Spectral4

import folium 
from folium import plugins
from folium.plugins import HeatMap

from mpl_toolkits.basemap import Basemap

import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
from plotly.graph_objs import Scatter, Figure, Layout
cf.set_config_file(offline=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import gc

# Any results you write to the current directory are saved as output.
def load_data():
    train  = pd.read_csv("../input/train.csv", nrows=10_00_000)
    test = pd.read_csv("../input/test.csv", nrows = 10_00_000)
    return train,test
%%time
train, test = load_data()
print(train.shape)
train.head(5)
%%time
train.dtypes.value_counts().plot.bar(figsize=(12, 6))
%%time
train.isnull().sum().plot.bar(figsize=(20, 6))
sns.jointplot(x = train.fare_amount, y = train.index, data= train,size=8, ratio=6, color="#f47442")
plt.figure(figsize=(20,8))
sns.distplot(train.fare_amount,color="orange")
sns.pairplot(train.head(10000))
plt.figure(figsize=(20,8))
sns.barplot(x="passenger_count",y="fare_amount", data=train, hue_order=True)
# sns.lmplot(x='passenger_count', y='fare_amount',data=train,fit_reg=False)
describe = train.describe()
describe
fig, axarr = plt.subplots(2, 3, figsize=(20, 8))
describe['fare_amount'].plot.bar(ax=axarr[0][0])
describe['pickup_longitude'].plot.bar(ax=axarr[0][1])
describe['pickup_latitude'].plot.bar(ax=axarr[0][2])
describe['dropoff_longitude'].plot.bar(ax=axarr[1][0])
describe['dropoff_latitude'].plot.bar(ax=axarr[1][1])
describe['passenger_count'].plot.bar(ax=axarr[1][2])
train.skew()
skew_data = train.skew()
plt.figure(figsize=(20,8))
sns.distplot(skew_data, bins=10, kde=True, color="orange")
plt.figure(figsize=(10,8))
sns.heatmap(train.corr(), annot=True)
latmin = 40.48
lonmin = -74.28
latmax = 40.93
lonmax = -73.65
ratio = np.cos(40.7 * np.pi/180) * (lonmax-lonmin) /(latmax-latmin)
from matplotlib.colors import LogNorm
fig = plt.figure(1, figsize=(20,15) )
hist = plt.hist2d(train.pickup_longitude,train.pickup_latitude,bins=199,range=[[lonmin,lonmax],[latmin,latmax]],norm=LogNorm())
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Pickup Locations')
plt.colorbar(label='Number')
plt.show()
fig = plt.figure(1, figsize=(20,15) )
hist = plt.hist2d(train.dropoff_longitude,train.dropoff_latitude,bins=199,range=[[lonmin,lonmax],[latmin,latmax]],norm=LogNorm())
plt.xlabel('Longitude [degrees]')
plt.ylabel('Latitude [degrees]')
plt.title('Dropoff Locations')
plt.colorbar(label='Number')
plt.show()
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
train = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]
train = train[(train.dropoff_longitude> xlim[0]) & (train.dropoff_longitude < xlim[1])]
train = train[(train.pickup_latitude> ylim[0]) & (train.pickup_latitude < ylim[1])]
train = train[(train.dropoff_latitude> ylim[0]) & (train.dropoff_latitude < ylim[1])]
longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)
latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)
plt.figure(figsize = (20,15))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)
plt.show()
loc_df = pd.DataFrame()
loc_df['longitude'] = longitude
loc_df['latitude'] = latitude
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

kmeans = KMeans(n_clusters=15, random_state=2, n_init = 10).fit(loc_df)
loc_df['label'] = kmeans.labels_

loc_df = loc_df.sample(200000)
plt.figure(figsize = (10,10))
for label in loc_df.label.unique():
    plt.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.3, markersize = 0.3)

plt.title('Clusters of New York')
plt.show()
fig,ax = plt.subplots(figsize = (10,10))
for label in loc_df.label.unique():
    ax.plot(loc_df.longitude[loc_df.label == label],loc_df.latitude[loc_df.label == label],'.', alpha = 0.4, markersize = 0.1, color = 'gray')
    ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')
    ax.annotate(label, (kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'b', fontsize = 20)
ax.set_title('Cluster Centers')
plt.show()
# Defining the box to work with
min_long = -74.25
max_long = -73.7
min_lat = 40.6
max_lat = 40.9

def filter_long(longi):
    return longi >= min_long and longi <= max_long

def filter_lat(lat):
    return lat >= min_lat and lat <= max_lat

train = train[(train['pickup_longitude'].apply(filter_long)) & (train['dropoff_longitude'].apply(filter_long))]
train = train[(train['pickup_latitude'].apply(filter_lat)) & (train['dropoff_latitude'].apply(filter_lat))]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20,10))
P_pickups = train.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',
                color='orange', xlim=(min_long,max_long), ylim=(min_lat, max_lat),
                s=.02, alpha=.6, subplots=True, ax=ax1)
ax1.set_title("Aggregate Pickups")
ax1.set_axis_bgcolor('black') #Background Color

P_dropoff = train.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',
                color='yellow', xlim=(min_long,max_long), ylim=(min_lat, max_lat),
                s=.02, alpha=.6, subplots=True, ax=ax2)
ax2.set_title("Aggregate DropOffs")
ax2.set_axis_bgcolor('black') #Background Color
# data = [go.Scattermapbox(
#             lat= train['pickup_latitude'] ,
#             lon= train['pickup_longitude'],
#             customdata = train['key'],
#             mode='markers',
#             marker=dict(
#                 size= 4,
#                 color = 'gold',
#                 opacity = .8,
#             ),
#           )]
# layout = go.Layout(autosize=False,
#                    mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
#                                 bearing=10,
#                                 pitch=60,
#                                 zoom=13,
#                                 center= dict(
#                                          lat=40.721319,
#                                          lon=-73.987130),
#                                 style= "mapbox://styles/shaz13/cjiog1iqa1vkd2soeu5eocy4i"),
#                     width=900,
#                     height=600, title = "Pick up Locations in NewYork")
# fig = dict(data=data, layout=layout)
# iplot(fig)
# data = [go.Scattermapbox(
#             lat= train['dropoff_latitude'] ,
#             lon= train['dropoff_longitude'],
#             customdata = train['key'],
#             mode='markers',
#             marker=dict(
#                 size= 4,
#                 color = 'cyan',
#                 opacity = .8,
#             ),
#           )]
# layout = go.Layout(autosize=False,
#                    mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
#                                 bearing=10,
#                                 pitch=60,
#                                 zoom=13,
#                                 center= dict(
#                                          lat=40.721319,
#                                          lon=-73.987130),
#                                 style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),
#                     width=900,
#                     height=600, title = "Drop off locations in Newyork")
# fig = dict(data=data, layout=layout)
# iplot(fig)
