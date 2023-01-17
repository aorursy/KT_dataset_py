# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data  = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]

city_location = pd.DataFrame(zip(data.neighbourhood,data.longitude,data.latitude))
gdf = gpd.GeoDataFrame(

    city_location, geometry=gpd.points_from_xy(data.longitude, data.latitude))



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



ax = world[world.name == 'NewYork'].plot(

    color='white', edgecolor='black')



gdf.plot(ax=ax, color='red')

plt.show()
