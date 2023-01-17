# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization scripts

# Maps

from mpl_toolkits.basemap import Basemap

# Plots

import matplotlib.pyplot as plt

from matplotlib import cm



# Cluster analysis

from sklearn.cluster import KMeans



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/SLC_Police_Cases_2016_cleaned_geocoded.csv')
df['reported'] = pd.to_datetime(df['reported'], errors='coerce')

df['occurred'] = pd.to_datetime(df['occurred'], errors='coerce')
df.head()
df['x_gps_coords'] = pd.to_numeric(df.x_gps_coords, errors='coerce')

df['y_gps_coords'] = pd.to_numeric(df.y_gps_coords, errors='coerce')

latlon = df[['x_gps_coords', 'y_gps_coords']]

latlon = latlon.dropna(axis=0)

kmeans = KMeans(n_clusters=10)

kmodel = kmeans.fit(latlon)

centroids = kmodel.cluster_centers_

clons, clats = zip(*centroids)
north = 41.0

south = 40.5

east = -111.7

west = -112.3

# Just get SLC-related GPS coordinates

gpsdfs = df[((df.x_gps_coords <= east) & (df.x_gps_coords >= west))

             & ((df.y_gps_coords >= south) & (df.y_gps_coords <= north))]

gpsdfs.describe()
fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i',

            ax=ax,

            area_thresh=1000.0)



x, y = m(gpsdfs.x_gps_coords.values, gpsdfs.y_gps_coords.values)

m.hexbin(x, y, cmap=cm.YlOrRd_r, bins='log', gridsize=1000)

cx, cy = m(clons, clats)

m.scatter(cx, cy, 3, color='red')
# Airport & city refs marked out because kaggle doesn't support them yet.

from bokeh.io import show

from bokeh.charts import output_notebook

from bokeh.plotting import figure

# from bokeh.sampledata import us_cities, airports

output_notebook()
# us_cities = us_cities.data.copy()

# airports = airports.data.copy()

# cities_xs = [us_cities[code]["lons"] for code in us_cities]

# cities_ys = [us_cities[code]["lats"] for code in us_cities]

# airports_xs = [airports[code]["lons"] for code in airports]

# airports_ys = [airports[code]["lats"] for code in airports]

p = figure(title="SLC Crime Reports and Centroids", 

           toolbar_location="left", plot_width=1100, plot_height=700)

# p.patches(cities_xs, cities_ys, fill_alpha=0.0,

#     line_color="#884444", line_width=0.8)

# p.patches(airport_xs, airport_ys, fill_alpha=0.0,

#     line_color="#884444", line_width=0.5)

p.circle(gpsdfs.x_gps_coords.data, gpsdfs.y_gps_coords.data, size=1, color='navy', alpha=0.3)

p.circle(clons, clats, size=5, color='red', alpha=0.6)

show(p)
# For better mapping, you want to use folium

# import folium