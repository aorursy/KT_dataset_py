# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import geopy.distance



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
estate = pd.read_csv("../input/sao-paulo-real-estate-sale-rent-april-2019/sao-paulo-properties-april-2019.csv")

metro = pd.read_csv("../input/sao-paulo-metro/metrosp_stations.csv", index_col="station")
mask = (estate.Longitude < -46) & (estate.Longitude > -46.9) & (estate.Latitude < -23) & (estate.Latitude > -24)

estate = estate[mask]
estate.columns = estate.columns.str.lower().str.replace(" ","_")
estate["property_coords"] = list(zip(estate.longitude, estate.latitude))

metro["station_coords"] = list(zip(metro.lon, metro.lat))
def get_list_of_nearest_stations(properties):

    nearest = []

    names = []

    distances = []

    for station, name in zip(metro.station_coords, metro.index):

        names.append(name)

        distances.append(geopy.distance.distance(properties, station).km)

        min_value_index = distances.index(min(distances))

    nearest.append(names[min_value_index])

    return pd.Series(nearest)





estate['nearest_metro_station'] = estate.property_coords.apply(get_list_of_nearest_stations)
sale = estate.negotiation_type == "sale"

estate_sale = estate[sale]

plt.figure(figsize=(10, 10))

sns.scatterplot(data=estate_sale, x="longitude", y="latitude", hue="price", size="price").set_title('For Sale')

sns.scatterplot(data=metro, x="lon", y="lat")
rent = estate.negotiation_type == "rent"

estate_rent = estate[rent]

plt.figure(figsize=(10, 10))

sns.scatterplot(data=estate_rent, x="longitude", y="latitude", hue="price", size="price").set_title('For Rent')

sns.scatterplot(data=metro, x="lon", y="lat")
estate_sale.groupby('nearest_metro_station')['price'].mean().sort_values().plot(kind='barh', figsize=(8,18), title="Mean sale prices per metro station area")
estate_rent.groupby('nearest_metro_station')['price'].mean().sort_values().plot(kind='barh', figsize=(8,18), title="Mean rent prices per metro station area")
estate_sale.price.describe().apply(lambda x: format(x, '.2f'))
estate_rent.price.describe().apply(lambda x: format(x, '.2f'))
estate_sale.corr()
estate_rent.nearest_metro_station.value_counts()