import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# Loading the Dataset

ffr = pd.read_csv('../input/fast-food-restaurants/FastFoodRestaurants.csv')

ffr = ffr.drop(ffr.columns[[3,9]],axis=1)  # dropping columns 3 and 9

ffr.head()
# checking for the highest number of restaurants

ffr['name'].value_counts()
# Visualisation Using GeoPandas & MatPlotLib

import matplotlib.pyplot as plt

import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon



%matplotlib inline
# In GeoPandas, you need to import a .shp file to plot on. You can find these kinds of files on Google!

state_map = gpd.read_file('../input/shape-files/states.shp')

fig,ax = plt.subplots(figsize = (15,15))

state_map.plot(ax = ax)
crs = {'init': 'epsg:4326'} #Coordinate Reference System

geometry = [Point(xy) for xy in zip( ffr["longitude"], ffr["latitude"])]

geometry [:3]
# Creating the Geo Dataset for plotting using GeoPandas

geo_ffr = gpd.GeoDataFrame(ffr, crs = crs, geometry = geometry)

geo_ffr.head()
# Visualisation for McDonalds

fig,ax = plt.subplots(figsize = (15,15))

state_map.plot(ax = ax, alpha = 0.4, color = "grey")

geo_ffr[geo_ffr['name'] == "McDonald's"].plot(ax=ax,  markersize=1, color = "black",  marker = "o", label = "McD")

plt.legend(prop={'size':15})



#Visualisation for Burger King

fig,ax = plt.subplots(figsize = (15,15))

state_map.plot(ax = ax, alpha = 0.4, color = "grey")

geo_ffr[geo_ffr['name'] == "Burger King"].plot(ax=ax, markersize=2, color = "blue", marker = "o", label = "BK")

plt.legend(prop={'size':15})