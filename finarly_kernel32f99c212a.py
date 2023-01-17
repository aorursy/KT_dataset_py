import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.head()
airbnb.info()
#it seems like last review and review per month has a lot of null values. host_name and name also have null figures.
#Last review data nulls seems to coincide with reviews per month data, is last review data important though? 

#Some hostings have only limited bookings and none of them left a review. or they've never had any reviews. 

#Because host_name is not needed, it shall be dropped. Also name column provides very little information and as it is too customised, also little use and will be dropped.

#Null values from last review shall be filled with unknown 
airbnb.fillna({'last_review':'unknown','reviews_per_month':0},inplace=True)
plt.figure(figsize=(12,8))

sns.heatmap(airbnb.isnull(),cmap='coolwarm',yticklabels=False)
airbnb.drop('name',axis=1,inplace=True)
nullhostids = airbnb[airbnb['host_name'].isnull()]['host_id'].values

nullhostids
airbnb[airbnb['host_id'].isin(nullhostids)]
#Looks like there are no other instnaces where the names of the hosts are given. since hostnames is pretty useless, itll be dropped
airbnb.drop('host_name',axis=1,inplace=True)
airbnb.isnull().sum()
#now data is clean of null values, lets start digging in
airbnb.head()
airbnb.groupby('neighbourhood_group')

#seems like Mahattan and Brooklyn are very popular, lets visualise this 
sns.set_style('whitegrid')

plt.figure(figsize=(10,8))

sns.countplot(data=airbnb,x='neighbourhood_group')
airbnb['neighbourhood'].nunique()

#lets just graph the top 10 and see from there
plt.figure(figsize=(13,10))

sns.countplot(data=airbnb,x='neighbourhood',order=airbnb['neighbourhood'].value_counts().head(8).index)

plt.tight_layout()

#it looks like after 8 there arent many changes, lets just show top 8 
#lets take a deeper dive into the neighbourhood_groups 
airbnb.groupby('neighbourhood_group')['price'].mean()
plt.figure(figsize=(10,10))

sns.violinplot(data=airbnb[airbnb['price'] < 1200],x='neighbourhood_group',y='price')
#we can see not only is Manhattan most popular, it is also on average the most expensive area. 
#lets see which area is most profitable

meanprice = airbnb.groupby('neighbourhood_group')['price'].mean()

group_counts = airbnb.groupby('neighbourhood_group')['id'].count()

group_counts
profit = meanprice * group_counts
profit

#we can see that Mahattan is by far most profitable 
profitdf = pd.DataFrame(profit)
profitdf.plot(kind='bar',figsize=(8,8),color='blue',legend=False)

plt.title('Neighbourhood Group Count')
airbnb['host_id'].value_counts().head(8)
plt.figure(figsize=(10,10))

sns.countplot(data=airbnb,x='host_id',order=airbnb['host_id'].value_counts().head(5).index)
top5hosts = airbnb['host_id'].value_counts().head(20).index
airbnb[airbnb['host_id'].isin(top5hosts)]['neighbourhood_group'].value_counts()

#for the top 20 hosts, pretty much all their listings are in Manhattan. 
airbnb.head()
import geopandas as gpd

import descartes

from shapely.geometry import Point,Polygon
#nyc_map = gpd.read_file('geo_export_b20d26db-09ac-41a0-b27b-4bc7d7d23818.shp')

#crs = {'init':'epsy:4326'}

#geometry = [Point(xy) for xy in zip(airbnb['longitude'], airbnb['latitude'])]
#geo_df = gpd.GeoDataFrame(airbnb,

#                         crs = crs,

#                         geometry = geometry)

#geo_df.head()
#sns.set_style('darkgrid')

#fig,ax = plt.subplots(figsize=(15,15))

#nyc_map.plot(ax=ax, alpha=1,color='green')

#noutlier = airbnb[airbnb['price'] < 350]['price']

#x = airbnb[airbnb['price'] < 350]['longitude']

#y = airbnb[airbnb['price'] < 350]['latitude']

#plt.scatter(x,y,c=noutlier,cmap='coolwarm')

#plt.title('NYC Price map')

#plt.colorbar()
airbnb.head()
plt.figure(figsize=(8,8))

airbnb['room_type'].value_counts().plot.bar(color='blue')

plt.title('Type of Room Popularity')

#we can see that entire houses or private rooms are far more popular than shared rooms
plt.figure(figsize=(12,8))

sns.boxplot(data=airbnb,y =airbnb[airbnb['price'] < 400]['price'],x=airbnb['room_type'])

#data looks as expected that whole houses will be the most expensive