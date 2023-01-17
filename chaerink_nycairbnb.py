# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sns
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', encoding='utf=8')

data.shape
data.head(3)
set(data['neighbourhood_group'])
set(data['room_type'])
plt.figure(figsize=(14,8))

plt.title("Price Distribution")

sns.distplot(data['price'])
data['price'].describe(percentiles = [0.2*x for x in range(5)])
before = len(data)

data = data[data['price']<=500]

after = len(data)

print("With price ceiling $500, {0} values were omitted, leaving {1}".format(before-after, after))
plt.figure(figsize=(14,8))

plt.title("Price Distribution - renewed")

sns.distplot(data['price'])
data['price'].describe(percentiles=[0.25*x for x in range(4)])
data.columns
data.groupby('neighbourhood_group')['price'].mean()
val = data.groupby('neighbourhood_group')['price'].mean()

ind = val.index



plt.figure(figsize=(13,7))

plt.bar(ind, val, color='khaki', edgecolor='k')

plt.title("AVG price among neighborhoods")

plt.ylabel('price($)')
manhattan = data[data['neighbourhood_group'] == 'Manhattan']

manhattan.sort_values(by='price', ascending=False).iloc[:5]
manhattan.groupby('neighbourhood')['price'].mean().sort_values(ascending=False)[:10]
manhattan.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)[:10]
from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



geometry = [Point(xy) for xy in zip(manhattan['longitude'], manhattan['latitude'])]

gdf = GeoDataFrame(manhattan, geometry=geometry)   



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(14, 8)), marker='^', color='red', markersize=15)

plt.xlim(manhattan.longitude.min()-5, manhattan.longitude.max()+5)

plt.ylim(manhattan.latitude.min()-5, manhattan.latitude.max()+5)