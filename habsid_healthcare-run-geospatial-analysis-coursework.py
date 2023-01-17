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
#reading the csv file in a pandas dataframe
path = ('../input/jogging-dataset/track.csv')
df = pd.read_csv(path)
# dataframe inspection
df.info()
#observing few entries
df.head()

#Inherenntly csv passes time as string type
df.dtypes

#conveting the time column from object time to timestamp
df = pd.read_csv(path, parse_dates=['time'])
df.dtypes
df.head()
# the time is in UTC , we will convert it to Jeruselum time where the person was jogging
# our data has regular index , we will change it to time based index for speed calculations
df.index
df.index = df['time']
df.index[:5]
#pandas by default have naive time stamp , we have to convert them to aware time stamp first and then convert it to right time zone
import pytz
#working on a single row before implementing on all the rows in the data frame
ts = df.index[0]
ts.tz_localize(pytz.UTC)
#converting to desired timezone
ts.tz_localize(pytz.UTC).tz_convert(pytz.timezone('Asia/Jerusalem'))
# Applying the timezone change to the entire index
df.index = df.index.tz_localize(pytz.UTC).tz_convert(pytz.timezone('Asia/Jerusalem'))
#examining the changed timezone
df.index[:5]
# Calculating speed 
#calculating distance from latitude and longitude points
# calculating time 
#"""Distance on a circle"""

import numpy as np


def circle_dist(lat1, lng1, lat2, lng2):
   # """
   # Distance on a circle (in km)

   # Parameters
   
   # lat1, lng1, lat2, lng2: float or array of float

   # Returns
   # distance:
   #   distance from ``(lat1, lng1)`` to ``(lat2, lng2)`` in kilometers.
    #"""
    phi1 = np.deg2rad(90 - lat1)
    phi2 = np.deg2rad(90 - lat2)

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)

    cos = (np.sin(phi1) * np.sin(phi2) * np.cos(theta1 - theta2) +
           np.cos(phi1) * np.cos(phi2))
    arc = np.arccos(cos)
    return arc * 6373  # Earth radius in km

#passing parameters to the circle_dist funtion to calculate distances
dist = circle_dist(df['lat'], df['lng'], df['lat'].shift(), df['lng'].shift())
#to observe the values in km
print('dist' ,dist[:3])
#to verify the distance in km
dist.sum()
# calculating time for every corresponding distance
dt= df['time']-df['time'].shift()
dt[:10]
#verifying the time calc
dt.sum()
#we need time in scalar quantity to get speed (currently its in timestamp format)= dist/time
#gives time in seconds and will again need conversion to hour format
dt.head()
print('in sec', dt[1].total_seconds())
print('in hour',dt[1].total_seconds()/3600)
#we can use timedelta64 from numpy to divide pandas time delta value to give us a scalar result and get values in km/hr
dt[1]/np.timedelta64(1,'h')
#verified from above calculation
df.head()

#here dist is in km and time is in seconds which is being converted to hour
speed = dist/(dt/np.timedelta64(1,'h'))
speed[:5]
#shows high variance, thus data needs to be resampled
#before resampling we add columns to the dataframe
df['dist'] = dist
df['dt'] = dt
df.head()
#Downsample the series into 1 minute bins and sum the values of the timestamps falling into a bin of 1 min i.e distance coverd in 1 min.
df1m = df.resample('1min').sum()
df1m.index[:5]
#notice the time along index, its now every min
#there is no dt column which is time differnce corresponding to index after data is resampled, its so because sum only works on scalar colums
df1m[:5]
print(df1m.columns)
df1m.head()
#it does not have dt column
# adding the dt column to the resampled dataframe df1m 
df['dt'] = dt/np.timedelta64(1,'h') #converting dt to scalar before resampling
df1m=df.resample('1min').sum()

#now calculating speed with the added dt column in resampled dataframe df1m
speed1m = df1m['dist']/df1m['dt']
speed1m[:10]
#statistics on speed
speed1m.describe()
#plotting charts using magic command to display charts along with code
%matplotlib inline
import matplotlib.pyplot as plt #matplotlib is used for visualisation
plt.rcParams['figure.figsize'] = (10,6) #setting display size of the plot
plt.style.use('seaborn-whitegrid')
speed1m.plot()
speed1m.plot.box()
import folium #to map the jogging route
m = folium.Map(location = [df['lng'].mean(), df['lat'].mean()] , zoom_start = 15 , tiles='Stamen Terrain') #tiles can be used to set type of map
m
#creating marker on the map with a specific location, color and size, having a pop text
row = df.iloc[32]
marker = folium.CircleMarker([row['lng'], row['lat']] , radius = 5, color = 'red', popup = 'Hey There!')
marker.add_to(m) #adding marker to map
m

#tracing the route on the map using the marker position to corresponding lat, lng with time . 
def add_marker(row): 
   marker = folium.CircleMarker([row['lng'], row['lat']] , radius = 5, color = 'red', popup = row['time'].strftime('%X')) #converting timestamps to string with strftime
   marker.add_to(m)
row = df.iloc[32]
add_marker(row)
m

#applying the add_marker to all the rows with apply
def add_marker(row): 
   marker = folium.CircleMarker([row['lng'], row['lat']] , radius = 5, color = 'red', popup = row['time'].strftime('%X'))
   marker.add_to(m)
df.apply(add_marker, axis = 1)
m
mdf = df.resample('T').mean() #resampling looses the time column as pandas cannot average time stamps , but we can use corresponding rowname which is a time stamp
def add_marker(row): 
   marker = folium.CircleMarker([row['lng'], row['lat']] , radius = 5, color = 'red', popup = row.name.strftime('%X'))
   marker.add_to(m)
mdf.apply(add_marker, axis = 1)
m
import shapely #to create a polygon area on the map
from shapely.geometry import Point, Polygon
pt = Point(1,2) # to create an array of points that will lie inside/outside the polygon area
pt.x
poly = Polygon([[0,0], [0,10] ,[10,10] ,[10,0]])
print('area', poly.area)
print('centroid',poly.centroid.xy)
print('poiny of intersection with pt:',poly.intersects(pt))
poly.intersects(Point(10,20))
#creating col of pts from long and lat columns of the dataframe
mdf['pt'] = mdf[['lng', 'lat']].apply(Point, axis = 1)
mdf.head()
#defining points of the polygon
mean_lng, max_lng = mdf['lng'].mean() , mdf['lng'].max()
mean_lat, max_lat = mdf['lat'].mean() , mdf['lat'].max()
#defining the polygon to see which points lie outside/inside
poly = Polygon([
    [mean_lng, mean_lat],
    [mean_lng, max_lat],
    [max_lng, max_lat],
    [max_lng, mean_lat]])
#boundary of polygon
poly.exterior.xy
#merging and transposing to get arrays of points with numpy
l = np.stack(poly.exterior.xy).T    
#converting back to regular array to pass through the PolyLine method
l_np=np.array(l)
l_list = l_np.tolist()
folium.PolyLine(l_list)

m = folium.Map(location = [df['lng'].mean(), df['lat'].mean()] , zoom_start = 15 , tiles='Stamen Terrain')
mdf = df.resample('T').mean() #resampling looses the time column as pandas cannot average time stamps , but we can use corresponding rowname which is a time stamp
mdf['pt'] = mdf[['lng', 'lat']].apply(Point, axis = 1)
def add_marker(row): 
   marker = folium.CircleMarker([row['lng'], row['lat']] , radius = 5, color = 'red', popup = row.name.strftime('%X'))
   marker.add_to(m)

m.add_child(folium.PolyLine((l_list), color ='yellow'))
mdf.apply(add_marker, axis = 1)
m

#highlighting points that fall inside the polygon to a different color
m = folium.Map(location = [df['lng'].mean(), df['lat'].mean()] , zoom_start = 15 , tiles='Stamen Terrain')
mdf = df.resample('T').mean() #resampling looses the time column as pandas cannot average time stamps , but we can use corresponding rowname which is a time stamp
mdf['pt'] = mdf[['lng', 'lat']].apply(Point, axis = 1)
def add_marker(row): 
    color = 'yellow' if poly.intersects(row['pt'])  else 'green'
    marker = folium.CircleMarker([row['lng'], row['lat']] , radius = 5,color= 'red' , popup = row.name.strftime('%X'),  fill=True, fill_opacity=0.7, fill_color=color)
    # fill= true and opacity valu is important as color may appear very light therwise
    marker.add_to(m)

m.add_child(folium.PolyLine((l_list), color ='yellow'))
mdf.apply(add_marker, axis = 1)
m






