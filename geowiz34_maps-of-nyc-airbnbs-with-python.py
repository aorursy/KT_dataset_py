#Import various python packages

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

from shapely import wkt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



plt.style.use('fivethirtyeight')
#Create a pandas dataframe of the Airbnb data

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



data.head(5)
#Review the data types

data.dtypes
#Review the columns 

data.columns
#Rename a column to accurately reflect Boroughs

data.rename(columns={'neighbourhood_group':'boroname'}, inplace=True)
#Review the listings by boroname

plt.figure(figsize=(10,10))

sns.scatterplot(x='longitude', y='latitude', hue='boroname',s=20, data=data)
#Get a count by borough

borough_count = data.groupby('boroname').agg('count').reset_index()
#Plot the count by borough

fig, ax1 = plt.subplots(1,1, figsize=(6,6)

                       )

sns.barplot(x='boroname', y='id', data=borough_count, ax=ax1)



ax1.set_title('Number of Listings by Borough', fontsize=15)

ax1.set_xlabel('Borough', fontsize=12)

ax1.set_ylabel('Count', fontsize=12)

ax1.tick_params(axis='both', labelsize=10)
#Here we are using geopandas to bring in a base layer of NYC boroughs

nyc = gpd.read_file(gpd.datasets.get_path('nybb'))

nyc.head(5)
#Rename the column to boroname, so that we can join the data to it on a common field

nyc.rename(columns={'BoroName':'boroname'}, inplace=True)

bc_geo = nyc.merge(borough_count, on='boroname')
#Plot the count by borough into a map

fig,ax = plt.subplots(1,1, figsize=(10,10))

bc_geo.plot(column='id', cmap='viridis_r', alpha=.5, ax=ax, legend=True)

bc_geo.apply(lambda x: ax.annotate(s=x.boroname, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)

plt.title("Number of Airbnb Listings by NYC Borough")

plt.axis('off')
#Now,lets take a look at the count by neighborhood. Use the file downloaded from https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas/cpf4-rkhq

nbhoods = pd.read_csv('../input/nbhoods/nynta.csv')

nbhoods.head(5)
#There is a lot going on here... first rename the column

nbhoods.rename(columns={'NTAName':'neighbourhood'}, inplace=True)



#Then, since this is a csv file, convert the geometry column text into well known text, this will allow you to plot its geometry correctly

nbhoods['geom'] = nbhoods['the_geom'].apply(wkt.loads)



#Now convert the pandas dataframe into a Geopandas GeoDataFrame

nbhoods = gpd.GeoDataFrame(nbhoods, geometry='geom')


#Lets take a look at what the neighborhoods look like

fig,ax = plt.subplots(1,1, figsize=(8,8))

nbhoods.plot(ax=ax)
#Lets get a count by neighborhood

nbhood_count = data.groupby('neighbourhood').agg('count').reset_index()
#Lets merge the spatial GeoPandas Dataframe (with geometry), with the nbhood_count layer that is aggregated

nb_count_geo = nbhoods.merge(nbhood_count, on='neighbourhood')

nb_count_geo.head(3)
#Lets take a look at the count by neighborhood

fig,ax = plt.subplots(1,1, figsize=(10,10))



base = nbhoods.plot(color='white', edgecolor='black', ax=ax)



nb_count_geo.plot(column='id', cmap='plasma_r', ax=base, legend=True)



plt.title("Number of Airbnb Listings by Neighborhood")

ax.text(0.5, 0.01,'White = No Data',

       verticalalignment='bottom', horizontalalignment='left',

       transform=ax.transAxes,

       color='blue', fontsize=12)

plt.axis('off')
#Create a point of each Airbnb location, and enable the "data" dataframe into a geopandas dataframe

data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))



#Now, do a spatial join... This code here runs an intersect analysis to find which neighborhood the Airbnb location is in

joined = gpd.sjoin(nbhoods, data, how='inner', op='intersects')
#Lets take a look 

joined.head(3)
#Drop the second geometry column

joined.drop(columns='geom', inplace=True)
#Rename the column. 

joined.rename(columns={'neighbourhood_left':'neighbourhood'}, inplace=True)



#Create a count of each neighborhood

nb_join_count = joined.groupby('neighbourhood').agg('count').reset_index()
#Get the "true count". Join this data to the original neighborhoods geometry 

true_count = nbhoods.merge(nb_join_count, on='neighbourhood')
#Lets plot the data

fig,ax = plt.subplots(1,1, figsize=(10,10))



base = nbhoods.plot(color='white', edgecolor='black', ax=ax)



true_count.plot(column='id',cmap='plasma_r', ax=base, legend=True)

plt.title('Number of Airbnb listings by Neighborhood in NYC')
#Create a data frame, and add data for Yankee stadium to it

yankee_stadium = pd.DataFrame()

yankee_stadium['name'] = ["Yankee Stadium"]

yankee_stadium['lon'] = -73.926186

yankee_stadium['lat'] = 40.829659

yankee_stadium
#Create a geodataframe of Yankee Stadium

yankee_stadium= gpd.GeoDataFrame(yankee_stadium, geometry=gpd.points_from_xy(yankee_stadium.lon, yankee_stadium.lat))
#Lets plot the data

fig,ax1 = plt.subplots(1,1, figsize=(10,10))

base = nbhoods.plot(color='orange',alpha=0.5, edgecolor='black', ax=ax1)

yankee_stadium.plot(markersize=300,ax=base)

plt.title('Yankee Stadium and NYC')
#Lets filter the neighborhoods down to Manahattan and the Bronx

man_bronx_geo = nbhoods.loc[(nbhoods['BoroName'] == 'Manhattan') | (nbhoods['BoroName'] == 'Bronx')]
#Plot Yankee Stadium with the Bronx and Manhattan

fig,ax = plt.subplots(1,1, figsize=(10,10))

yankee_stadium.plot(markersize=300,color='red',ax=ax)

man_bronx_geo.plot(column='BoroName', cmap = 'tab20b',alpha=.5, ax=ax, legend=True)

plt.title("Bronx, Manhattan, and Yankee Stadium")
#Create a pandas dataframe of the Airbnb data

subways = pd.read_csv('../input/nyc-subway-stations/DOITT_SUBWAY_STATION_01_13SEPT2010.csv')

subways.head(5)
#Then, since this is a csv file, convert the geometry column text into well known text, this will allow you to plot its geometry correctly

subways['geom'] = subways['the_geom'].apply(wkt.loads)



#Now convert the pandas dataframe into a Geopandas GeoDataFrame

subways = gpd.GeoDataFrame(subways, geometry='geom')
#Lets take a look at what the neighborhoods look like

fig,ax = plt.subplots(1,1, figsize=(8,8))

subways.plot(ax=ax)

yankee_stadium.plot(markersize=100,ax=ax)

plt.title('NYC Subway Stations and Yankee Stadium', fontsize=12)
subways = subways[subways['LINE'].str.contains('4') | (subways['LINE'].str.contains('D'))]
#Lets take a look at what the neighborhoods look like

fig,ax = plt.subplots(1,1, figsize=(8,8))

subways.plot(ax=ax)

yankee_stadium.plot(markersize=100,ax=ax)

plt.title('NYC Subway Stations Servicing 4 and D lines, with Yankee Stadium', fontsize=10)
#Plot the count by borough into a map

fig,ax = plt.subplots(1,1, figsize=(10,10))



yankee_stadium.plot(markersize=300,color='red',ax=ax, label='Yankee Stadium')



subways.plot(markersize=50, color='green',ax=ax, label='Subways')



man_bronx_geo.plot(column='BoroName', cmap = 'tab20b',alpha=.5, ax=ax, legend=True)



plt.title("Bronx, Manhattan, and Yankee Stadium")



handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='center right')
#2 miles in feet is .001 * 32.195122 

yankee_stadium.crs = {'init' :'epsg:2263'}

stadium_buff = yankee_stadium.buffer(.001 * 32.195122)
stadium_buff = gpd.GeoDataFrame({'geometry': stadium_buff})
stadium_buff
stadium_buff.crs = {'init' :'epsg:2263'}

data.crs = {'init' :'epsg:2263'}
airbnbs_within_2m_of_ys = gpd.sjoin(data,stadium_buff, how='inner', op='intersects')
len(airbnbs_within_2m_of_ys)
#Plot the airbnbs within 2 miles of Yankee stadium

fig,ax = plt.subplots(1,1, figsize=(10,10))

airbnbs_within_2m_of_ys.plot(markersize=50,ax=ax, label="Airbnbs")

yankee_stadium.plot(markersize=300,color='red',ax=ax, label="Yankee Stadium")

#man_bronx_geo.plot(column='BoroName', cmap = 'tab20b',alpha=.5, ax=ax, legend=True)

plt.title("Airbnbs within 2 miles of Yankee Stadium")



handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='upper right')
#Lets add our crieria, one by one, and see how many listings are left after each

print("Starting number of airbnbs: {0}".format(len(airbnbs_within_2m_of_ys)))



airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['price'] < 250]

print("Number of airbnbs after cutting price to less than $250: {0} ".format(len(airbnbs_within_2m_of_ys)))



airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['availability_365'] > 240]

print("Number of airbnbs after selecting those that are available at least 6 months out of the year: {0} ".format(len(airbnbs_within_2m_of_ys)))



airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['number_of_reviews'] >= 10]

print("Number of airbnbs after selecting those with at least 10 reviews: {0} ".format(len(airbnbs_within_2m_of_ys)))



airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['room_type'] == 'Entire home/apt']

print("Number of airbnbs after selecting those that offer the entire home/apt: {0} ".format(len(airbnbs_within_2m_of_ys)))



airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['minimum_nights'] <= 3]

print("Number of airbnbs left after selecting airbnbs that have a minimum night stay of 3 or less: {0} ".format(len(airbnbs_within_2m_of_ys)))
airbnbs_within_2m_of_ys.head()
#Plot the airbnbs within 2 miles of Yankee stadium

fig,ax = plt.subplots(1,1, figsize=(10,10))

airbnbs_within_2m_of_ys.plot(markersize=100,ax=ax, legend=True, label="Airbnbs")

yankee_stadium.plot(markersize=300,color='red',ax=ax, label="Yankee Stadium")

plt.title("Airbnbs' near Yankee Stadium that meet our criteria so far")



handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='upper right')
#1/4 mile in feet is .001 * 3.657 

subways.crs = {'init' :'epsg:2263'}

subways_buff = subways.buffer(.001 * 3.657)
#Create a geodataframe for the subways buffer. Set the crs to 2263

subways_buff = gpd.GeoDataFrame({'geometry': subways_buff})

subways_buff.crs = {'init' :'epsg:2263'}
#Rename the index_right column. It can not be in our final spatial join

airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.rename(columns={'index_right': 'other_name'})
#Lets find the airbnbs that intersect our subway buffers

final_abs = gpd.sjoin(airbnbs_within_2m_of_ys,subways_buff, how='inner', op='intersects')
#How many airbnbs meet all of our conditions?

print("There are {0} airbnbs that meet all of the conditions".format(len(final_abs)))
#Plot the final results

fig,ax = plt.subplots(1,1, figsize=(10,10))

final_abs.plot(markersize=100,ax=ax, label="Airbnbs")

yankee_stadium.plot(markersize=300,color='red',ax=ax, label="Yankee Stadium")



plt.title("Airbnbs that meet all conditions")



handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='upper right')
#Lets look at the final results

final_abs