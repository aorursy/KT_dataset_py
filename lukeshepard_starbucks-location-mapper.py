# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import geopandas

import descartes

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/store-locations/directory.csv")
df[:5]
df[df["City"] == "Chicago"]
import geopandas
streets = geopandas.read_file("/kaggle/input/chicago-street-center-lines/geo_export_19287a01-62a7-454b-ac57-3cf755f2a986.shp")
fix,ax = plt.subplots(figsize=(15,15))

streets.plot(ax = ax)
from shapely.geometry import Point, Polygon

chicago_starbucks = df[df["City"] == "Chicago"]

geometry = [Point(xy) for xy in zip(chicago_starbucks["Longitude"], chicago_starbucks["Latitude"])]
crs = {'init': 'epsg:4326'}

gdf = geopandas.GeoDataFrame(chicago_starbucks, crs=crs, geometry=geometry)
fix,ax = plt.subplots(figsize=(15,15))

streets.plot(ax = ax, alpha=0.4, color="grey")

gdf.plot(ax = ax, markersize=20, color="green", label="Starbucks")

plt.legend(prop={'size':15})
def closest_distance(row):

    p = Point(row["Longitude"], row["Latitude"])

    

    return min([p.distance(p2) for p2 in geometry if p2 != p])



closest_distance(chicago_starbucks.iloc[5])
chicago_starbucks["closest_distance"] = chicago_starbucks.apply(closest_distance, axis=1)
fix,ax = plt.subplots(figsize=(15,15))

streets.plot(ax = ax, alpha=0.4, color="grey")

gdf.plot(ax = ax, markersize=20, cmap='winter', alpha=1.0, column='closest_distance', legend=True, label="Starbucks")

plt.legend(prop={'size':15})
gdf_plot = gdf[gdf['closest_distance'] < 0.01]



fix,ax = plt.subplots(figsize=(30,30))

streets.plot(ax = ax, alpha=0.4, color="grey")

gdf_plot.plot(ax = ax, markersize=20, cmap='winter', alpha=1.0, column='closest_distance', legend=True, label="Starbucks")

plt.legend(prop={'size':15})
gdf[gdf['closest_distance'] < 0.01]['closest_distance'].hist()
gdf[gdf['closest_distance'] < 0.01]