!pip install earthpy
import os

import matplotlib.pyplot as plt

import numpy as np

import rasterio as rio

import geopandas as gpd

import earthpy as et

import earthpy.spatial as es

import earthpy.plot as ep
# Downloading data using below code

data = et.data.get_data('cold-springs-fire')
# Let's set the working directory

os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))
# Defining the path of the '.tif' file to be used

naip_data_path = os.path.join("data", "cold-springs-fire", 

                              "naip", "m_3910505_nw_13_1_20150919", 

                              "crop", "m_3910505_nw_13_1_20150919_crop.tif")
naip_data_path
# opening the data using rasterio module as shown below

with rio.open(naip_data_path) as src:

    naip_data = src.read()
naip_data.shape
# Calculating NDVI by considering 1st and 4th layer 

naip_ndvi = es.normalized_diff(naip_data[3], naip_data[0])
naip_ndvi
# Let's now visualize the NDVI values using earthpy.plot submodule

ep.plot_bands(naip_ndvi, cmap='RdYlGn', scale=False, vmin=-1, vmax=1,

              title="Analyzing calculated NDVI values")

plt.show()
# Plotting a histogram will be much more informative and accurate.

ep.hist(naip_ndvi, colors='lightblue' ,figsize=(12, 6),

        title=["Visualizing Distribution of NDVI values"])



plt.show()