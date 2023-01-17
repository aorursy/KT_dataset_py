# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Constants

INPUT_DIR = '/kaggle/input'

# Any results you write to the current directory are saved as output.

OUTPUT_DIR = './'

DATASET = 'crash_data.csv'
# Load dataset into dataframe

dataframe = pd.read_csv(os.path.join(INPUT_DIR, DATASET))

dataframe.shape
# Show dataframe

dataframe.head()
# Create a geodataframe from dataframe

geodataframe = gpd.GeoDataFrame(dataframe, geometry=gpd.points_from_xy(dataframe.lng, dataframe.lat))

geodataframe.shape
# Show geodataframe

geodataframe.head()
# Plot the coordinates over a country-level map

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



# We restrict to United States of America

ax = world[world.name == 'United States of America'].plot(

    color='white', edgecolor='black')



# We can now plot our GeoDataFrame

geodataframe.plot(ax=ax, color='red')



# Save PNG image

plt.savefig(os.path.join(OUTPUT_DIR, 'crash_data.png'))



# Plot GeoDataFrame

plt.show()
# Save GeoJSON from GeoDataFrame

geodataframe.to_file(os.path.join(OUTPUT_DIR, 'crash_data.geojson'), driver='GeoJSON')