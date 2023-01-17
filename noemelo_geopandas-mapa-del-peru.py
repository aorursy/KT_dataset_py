%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import geopandas as gpd
url_geojson = "/kaggle/input/geojson-departamentos-peru/peru_departamental_simple.geojson"
region_geojson = gpd.read_file(url_geojson)

region_geojson.head()
ax = region_geojson.plot(figsize=(15,15))

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.show()
ax = region_geojson.plot(figsize=(20,20),edgecolor=u'gray', cmap='Pastel1')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.show()
url_geojson = "/kaggle/input/geojson-departamentos-peru/peru_provincial_simple.geojson"



provinces_geojson = gpd.read_file(url_geojson)

provinces_geojson.head()
ax = provinces_geojson.plot(figsize=(20,20))

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.show()
ax = provinces_geojson.plot(figsize=(20,20),edgecolor=u'gray', cmap='Pastel1')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.show()
ax = provinces_geojson[provinces_geojson.FIRST_NOMB=='CUSCO'].plot(figsize=(20,20),edgecolor=u'gray', cmap='Pastel1')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

ax.axis('scaled')

plt.show()
url_geojson = "/kaggle/input/geojson-departamentos-peru/peru_distrital_simple.geojson"



districts_geojson = gpd.read_file(url_geojson)

districts_geojson.head()
ax = districts_geojson.plot(figsize=(20,20))

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.show()
ax = districts_geojson.plot(figsize=(20,20),edgecolor=u'gray', cmap='Pastel1')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

plt.show()
ax = districts_geojson[districts_geojson.NOMBDEP=='CUSCO'].plot(figsize=(20,20),edgecolor=u'gray', cmap='Pastel1')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

ax.axis('scaled')

plt.show()
ax = districts_geojson[districts_geojson.NOMBPROV=='CALCA'].plot(figsize=(20,20),edgecolor=u'gray', cmap='Pastel1')

plt.ylabel('Latitude')

plt.xlabel('Longitude')

ax.axis('scaled')

plt.show()
"""

Archivos:

"""

# /kaggle/input/geojson-departamentos-peru/peru_distrital_simple.geojson

# /kaggle/input/geojson-departamentos-peru/peru_provincial_simple.geojson

# /kaggle/input/geojson-departamentos-peru/peru_departamental_simple.geojson