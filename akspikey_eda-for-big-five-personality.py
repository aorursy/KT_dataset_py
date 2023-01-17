import pandas as pd

from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

%matplotlib inline
data_set = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
data_set = data_set.dropna()
lat_cor = pd.to_numeric(data_set['lat_appx_lots_of_err'], errors='coerce')

long_cor = pd.to_numeric(data_set['long_appx_lots_of_err'], errors='coerce')
geometry = [Point(xy) for xy in zip(long_cor, lat_cor)]
gdf = GeoDataFrame(data_set, geometry=geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(40, 6)), marker='o', color='red', markersize=5);
extract_country = data_set[['country', 'IPC']]
country_group = extract_country.groupby('country').agg('count').reset_index()
plt.figure(figsize=(15, 6))

ax = sns.barplot(x='country', y='IPC', data=country_group.sort_values('IPC',ascending=False).head(50))

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15, 6))

ax = sns.barplot(x='country', y='IPC', data=country_group.sort_values('IPC',ascending=False).tail(50))

plt.xticks(rotation=90)

plt.show()
data_set['year'] = data_set.apply(lambda x: datetime.strptime(x['dateload'], '%Y-%m-%d %H:%M:%S').year, axis=1)

data_set['month'] = data_set.apply(lambda x: datetime.strptime(x['dateload'], '%Y-%m-%d %H:%M:%S').month, axis=1)
extract_date = data_set[['year', 'month', 'IPC']]
date_analysis = pd.DataFrame(extract_date.groupby(['year', 'month']).agg('count')).reset_index()
plt.figure(figsize=(15, 6))

sns.barplot(x="year", hue="month", y="IPC", data=date_analysis)

plt.show()