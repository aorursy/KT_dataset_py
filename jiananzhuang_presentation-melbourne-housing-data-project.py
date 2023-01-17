# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import HTML, display,Image
import json
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# To process data
import shapely
from shapely.geometry import Point
import unicodedata
import pysal as ps
import folium
from folium.plugins import MarkerCluster
from folium.map import *
from branca.colormap import linear
import matplotlib.units as units
import matplotlib.dates as dates
import matplotlib.ticker as ticker
import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/melbourne-housing-market"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
#read data from Kaggle
df = pd.DataFrame([data])
data.fillna(0)
#get all Nan data to be replaced by 0
#Basic plot to view the distribution of data
data.hist(bins=50, figsize=(20,20))
plt.show()
#bins give number of data taking into graph, and figsize is the dimension of the data
#x is number data distribution, and y is the value
# Get Average price for Distance between CBD below 1km
Distance_below_1km = data['Distance'] < 1
Distance1 = data[Distance_below_1km].mean()
Distance1['Price']
df1 = pd.DataFrame(data=Distance1)
df1.columns = ['Distance below 1km']
df1

# Get Average price for Distance between CBD below 2km and above 3km
Distance_below_2km = data['Distance'] < 2
Distance_above_1km = data['Distance'] > 1
Distance_1_2 = Distance_below_2km & Distance_above_1km

Distance12 = data[Distance_1_2].mean()
Distance12['Price']
df12 = pd.DataFrame(data=Distance12)
df12.columns = ['Distance between 1 - 2 km']
df12
# Get Average price for Distance between CBD below 2km and above 3km
Distance_below_3km = data['Distance'] < 3
Distance_above_2km = data['Distance'] > 2
Distance_2_3 = Distance_below_3km & Distance_above_2km

Distance23 = data[Distance_2_3].mean()
Distance23['Price']

df23 = pd.DataFrame(data=Distance23)
df23.columns = ['Distance between 2 - 3 km']
df23
# Get Average price for Distance between CBD below 3km and above 4km
Distance_below_4km = data['Distance'] < 4
Distance_above_3km = data['Distance'] > 3
Distance_3_4 = Distance_below_4km & Distance_above_3km

Distance34 = data[Distance_3_4].mean()
Distance34['Price']

df34 = pd.DataFrame(data=Distance34)
df34.columns = ['Distance between 3 - 4 km']
df34
# Get Average price for Distance between CBD below 4km and above 5km
# At the same time, contruction new data frame for matplotlid later in analysis
Distance_below_5km = data['Distance'] < 5
Distance_above_4km = data['Distance'] > 4
Distance_4_5 = Distance_below_5km & Distance_above_4km

Distance45 = data[Distance_4_5].mean()

df45 = pd.DataFrame(data=Distance45)
df45.columns = ['Distance between 4 - 5 km']
df45
#Add all data above together
raw_result = pd.concat([df45, df34, df23, df12, df1], axis=1, join_axes=[df45.index])
raw_result
result = raw_result.drop(['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount'])
result
data.plot(kind='scatter', x='Distance', y='Price',alpha = 1,color = 'blue')
plt.xlabel('Distance(km)')              # label = name of label
plt.ylabel('Price(Million)')
plt.title('Pricing vs Distance')            # title = title of plot
#change x label
data.plot(kind="scatter", x="Longtitude", y="Lattitude",
    s = data["Landsize"]/100, label="Landsize",
    c="Price", cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.4, figsize=(10,10),
)
plt.legend()
plt.show()
ax = result[['Distance below 1km', 'Distance between 1 - 2 km', 'Distance between 2 - 3 km', 'Distance between 3 - 4 km','Distance between 4 - 5 km']].plot(kind='bar', title ="Price vs Distance to CBD", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Label", fontsize=12)
ax.set_ylabel("Price(Million)", fontsize=12)
plt.show
map_distance_vs_price = folium.Map(location=[-37.81, 144.96],
                        zoom_start = 12)
#Uni Point
folium.Marker([-37.7963, 144.9614],
              popup='Melbourne University',
              icon=folium.Icon(color='red',icon='cloud')
              ).add_to(map_distance_vs_price)

folium.Marker([-37.80782, 144.96294],
              popup='Melbourne RMIT',
              icon=folium.Icon(color='red',icon='cloud')
              ).add_to(map_distance_vs_price)

folium.Marker([-37.91407, 145.13170],
              popup='Monash University',
              icon=folium.Icon(color='red',icon='cloud')
              ).add_to(map_distance_vs_price)

#AIRPORT Point
folium.Marker([-37.66974, 144.84881],
              popup='AIRPORT',
              icon=folium.Icon(color='green')
              ).add_to(map_distance_vs_price)

# Circle marker
folium.Circle([-37.81, 144.96],
                    radius=5000,
                    popup='Price: $1,156,086',
                    color='#602320',
                    fill=True,
                    fill_color='#602320',
                    fill_opacity=0.01
                   ).add_to(map_distance_vs_price)

folium.Circle([-37.81, 144.96],
                    radius=4000,
                    popup='Price: $1,237,823',
                    color='#a32020',
                    fill=True,
                    fill_color='#a32020',
                    fill_opacity=0.01
                   ).add_to(map_distance_vs_price)

folium.Circle([-37.81, 144.96],
                    radius=3000,
                    popup='Price: $1,059,587',
                    color='#e0301e',
                    fill=True,
                    fill_color='#e0301e',
                    fill_opacity=0.01
                   ).add_to(map_distance_vs_price)

folium.Circle([-37.81, 144.96],
                    radius=2000,
                    popup='Price: $1,120,940',
                    color='#eb8c00',
                    fill=True,
                    fill_color='#eb8c00',
                    fill_opacity=0.01
                   ).add_to(map_distance_vs_price)

folium.Circle([-37.81, 144.96],
                    radius=1000,
                    popup='Price:$ 810,839.50',
                    color='#dc6900',
                    fill=True,
                    fill_color='#dc6900',
                    fill_opacity=0.01
                   ).add_to(map_distance_vs_price)


# colour selection from http://www.color-hex.com/color-palette/1217

map_distance_vs_price
PieDistance1 = data[Distance_below_1km]
PieDistance1
PieDistance12 = data[Distance_1_2]
PieDistance12
PieDistance23 = data[Distance_2_3]
PieDistance23
PieDistance34 = data[Distance_3_4]
PieDistance34
PieDistance45 = data[Distance_4_5]
PieDistance45
#To calculation frequency of Rooms a appear in data
counts1 = PieDistance1['Rooms'].value_counts(normalize=True)
counts1
df = pd.DataFrame(counts1)
newcounts1 = df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
newcounts1
counts12 = PieDistance12['Rooms'].value_counts(normalize=True)
counts12
df = pd.DataFrame(counts12)
newcounts12 = df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
newcounts12
counts23 = PieDistance23['Rooms'].value_counts(normalize=True)
counts23
df = pd.DataFrame(counts23)
newcounts23 = df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
newcounts23
counts34 = PieDistance34['Rooms'].value_counts(normalize=True)
counts34
counts45 = PieDistance45['Rooms'].value_counts(normalize=True)
counts45
df = pd.DataFrame(counts45)
newcounts45 = df.sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, by=None)
newcounts45
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels1 = 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bedroom 4'
sizes1 = newcounts1
fig1, ax1 = plt.subplots()

ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distance below 1 km')
plt.show()
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels12 = 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bedroom 4', 'Bedroom 5', 'Bedroom 6'
sizes12 = newcounts12
fig1, ax1 = plt.subplots()

ax1.pie(sizes12, labels=labels12, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distance between 1 - 2 km')
plt.show()

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels23 = 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bedroom 4', 'Bedroom 5', 'Bedroom 6', 'Bedroom 7', 'Bedroom 8', 'Bedroom 10'
sizes23 = newcounts23
fig1, ax1 = plt.subplots()

ax1.pie(sizes23, labels=labels23, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distance between 2 - 3 km')
plt.show()

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels34 = 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bedroom 4', 'Bedroom 5', 'Bedroom 6', 'Bedroom 7'
sizes34 = counts34
fig1, ax1 = plt.subplots()

ax1.pie(sizes34, labels=labels34, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distance between 3 - 4 km')
plt.show()


labels45 = 'Bedroom 1', 'Bedroom 2', 'Bedroom 3', 'Bedroom 4', 'Bedroom 5', 'Bedroom 6', 'Bedroom 9', 'Bedroom 12'
sizes45 = newcounts45
fig1, ax1 = plt.subplots()

ax1.pie(sizes45, labels=labels45, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Distance between 4 - 5 km')
plt.show()



