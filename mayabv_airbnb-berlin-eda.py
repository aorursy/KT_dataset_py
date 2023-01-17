import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from geopy.distance import great_circle

import matplotlib.ticker as ticker

import os
df = pd.read_csv("../input/airbnb-berlin/listings.csv")

df.head()
df.describe()
#To see what data types we have

df.info()
#To detect the missing data



df.isnull().sum()
#To visualise the missing values

fig = plt.figure(figsize=(12,8))

sns.heatmap(df.isnull())

plt.show()
#Dealing with missing values

#For the sake of experiment I will remove a couple of price values from the df slice

df3 = df[['neighbourhood','room_type','price']]

df_missingvalues = df3[df3.neighbourhood =='Alexanderplatz'].head(10)

df_missingvalues.loc[48:53,'price'] = 0

df_missingvalues



#use the average value (we work with a slice of the data here)



df_missingvalues[df_missingvalues.room_type == "Entire home/apt"].mean()
#replasing the data

#location based imputation

df_missingvalues.loc[48,'price'] = 48.5

df_missingvalues
#fill in all missing values simultaneusly with a single value 



df_missingvalues.loc[48:53,'price'] = np.nan 

mean = df_missingvalues['price'].mean() #(use of mean or median would depend on the data and presence of outliers)

df_missingvalues['price'].fillna(mean, inplace=True)

df_missingvalues
#Predictive filling using the interpolate() method will perform a linear interpolation to 'find' the missing values

df_missingvalues.loc[48:53,'price'] = np.nan 

df_missingvalues['price'].interpolate(inplace=True)

df_missingvalues
df['room_type'].value_counts()
df['neighbourhood_group'].value_counts().head()
#Using box plots to analyse our data



fig = plt.figure(figsize=(12,8))

sns.boxplot(x='room_type',y='price',data=df, showfliers=False);
#Adding a Distance from centre column



def distance_from_centre(lat, lon):

    berlin_centre = (52.520008, 13.404954)

    apartment_spot = (lat, lon)

    return round(great_circle(berlin_centre, apartment_spot).km, 1)



df["distance"] = df.apply(lambda x: distance_from_centre(x.latitude, x.longitude), axis=1)

df.head()
fig = plt.figure(figsize=(12,8))

plt.scatter(df['distance'],df['price'])

plt.xlim(0,30)

plt.ylim(0,6500)

plt.xlabel('Distance from centre')

plt.ylabel('Price')

plt.show()
f,ax = plt.subplots(figsize=(17,8))

ax = sns.swarmplot(y= df.index,x= df.minimum_nights)

n = 5



for index, label in enumerate(ax.xaxis.get_ticklabels(), 1):

    if index % n != 0:

        label.set_visible(False)



plt.show()
fig = plt.figure(figsize=(12,8))

plt.hist(df['price'], bins=1000)

plt.xlim(0,300)

plt.xlabel('Price')

plt.ylabel('Number of properties')

plt.grid()

plt.show()
#To find the average price per neighbourhood



df1 = df[['neighbourhood_group','price', 'distance']]

df_group = df1.groupby(['neighbourhood_group'],as_index=False).mean()

df_group.head(15)
#Or group by location and property type to see the average price per each type

df2 = df[['neighbourhood_group','room_type','price']]

df_group2 = df2.groupby(['neighbourhood_group','room_type'],as_index=False).mean()

df_group2.head()
#The above can be converted to pivot

df_pivot = df_group2.pivot(index='neighbourhood_group',columns='room_type')

df_pivot.head()
df_var = df1[(df1['neighbourhood_group'] == 'Pankow') | (df1['neighbourhood_group'] == 'Mitte')]

df_vartemp = df_var[['neighbourhood_group','price']].groupby(['neighbourhood_group'])

stats.f_oneway(df_vartemp.get_group('Pankow')['price'],df_vartemp.get_group('Mitte')['price'])
import folium

from folium import plugins

from folium.plugins import HeatMap
map_test = df.head(200)
m = folium.Map(location=[52.52, 13.4], zoom_start = 12)



heat_data = [[row['latitude'],row['longitude']] for index, row in

             map_test[['latitude', 'longitude']].iterrows()]



hh =  HeatMap(heat_data).add_to(m)



m
f,ax = plt.subplots(figsize=(12,8))

ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.availability_365)

plt.show()