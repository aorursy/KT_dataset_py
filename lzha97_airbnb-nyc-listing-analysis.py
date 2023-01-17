# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

import descartes

from shapely.geometry import Point, Polygon

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

plt.style.use('Solarize_Light2')



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ABdata = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data_head = ABdata.head()

data_head
ABdata.dtypes
ABdata.isnull().sum()
(ABdata.number_of_reviews == 0).sum()
print("Data Dimensions:", ABdata.shape, '\n')

print(len(ABdata.id.unique()), 'unique listing ids.')

print(len(ABdata.host_id.unique()), "unique host ids.")

print(len(ABdata.neighbourhood.unique()), "unique neighbourhoods.")



print("\nRoom Types:")

print(ABdata.room_type.value_counts())

zero_price = ABdata[ABdata.price == 0]

print(zero_price.shape)

zero_price
neighbourhood_group_freq = ABdata.neighbourhood_group.value_counts()

print(neighbourhood_group_freq)

ng_freq_plot = neighbourhood_group_freq.plot.bar(title="Neighbourhood group frequency", color = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'])
crs= {'init': 'epsg:4326'}

street_map = gpd.read_file('/kaggle/input/nyu-shapefile/nyu_2451_34490.shp')

geometry = [Point(xy) for xy in zip(ABdata.longitude, ABdata.latitude)]

geometry[0:5]

geo_df = gpd.GeoDataFrame(ABdata, crs=crs, geometry=geometry)

geo_df.drop(columns=['latitude','longitude'])

geo_df.head(2)
#plot the locations of the listings, colored by the room type

fig, ax = plt.subplots(figsize=(10,10))

street_map.plot(ax = ax, alpha = 0.4, color = "grey")



geo_df[geo_df['room_type'] == 'Private room'].plot(ax=ax, markersize = 5, color = "red", marker = 'o', label = "private room")

geo_df[geo_df['room_type'] == 'Entire home/apt'].plot(ax=ax, markersize = 5, color = "blue", marker = '^', label = "entire apartment")

geo_df[geo_df['room_type'] == 'Shared room'].plot(ax=ax, markersize = 5, color = "yellow", marker = '*', label = "shared room")

plt.legend(prop={'size': 15})
data = [['Timesquare', 40.757918, -73.985489], ['Metropolitan Museum of Art', 40.779242, -73.962665], ['Empire State', 40.748432, -73.985557], ['Chinatown', 40.717717, -73.995995]]

pop_dest = df = pd.DataFrame( data, columns = ['name', 'latitude', 'longitude']) 

print(pop_dest)

fig, ax = plt.subplots(figsize=(10,10))

street_map.plot(ax = ax, alpha = 0.4, color = "grey")

geo_df[geo_df['room_type'] == 'Shared room'].plot(ax=ax, markersize = 5, color = "yellow", marker = '*', label = "shared room")

plt.legend(prop={'size': 15})

ax.scatter(x=pop_dest.longitude, y = pop_dest.latitude)
# Find the outliers for price and reviews per month (no output displayed)



quart25 = geo_df.price.quantile(0.25) 

quart75 = geo_df.price.quantile(0.75)

inter_quart = quart75 - quart25



min_val = quart25 - 1.5*inter_quart

max_val = quart75 + 1.5*inter_quart



outliers_gone = geo_df[(geo_df.price > min_val) & (geo_df.price < max_val)]

outliers_gone.head()

outliers_gone.shape



outliers = geo_df[geo_df.price > max_val]



quart999 = geo_df.price.quantile(0.999)

top1percent = geo_df[geo_df.price >= quart999]
sns.set_palette('husl')

plt.figure(figsize=(15,10))

scatter = sns.violinplot(x='room_type', y='price', data=outliers_gone)
# plot listings colored by price 

f = plt.figure(figsize=(28, 10))



ax1 = f.add_subplot(121)

ax2 = f.add_subplot(122)



street_map.plot(ax = ax1, alpha = 0.4, color = "grey")

street_map.plot(ax = ax2, alpha = 0.4, color = "grey")

#street_map.plot(ax = ax3, alpha = 0.4, color = "grey")



pts = ax1.scatter(outliers_gone.longitude, outliers_gone.latitude, marker='o', s = 3, c=outliers_gone.price)

cbar1 = f.colorbar(pts, ax=ax1)

cbar1.set_label('price')

ax1.set_title("Airbnb Pricing Without Outliers")





pts2 = ax2.scatter(outliers.longitude, outliers.latitude, marker='o', s = 3, c=outliers.price)

cbar2 = f.colorbar(pts2, ax=ax2)

cbar2.set_label('price')

ax2.set_title("Airbnb Pricing of Outliers")



'''

pts3 = ax3.scatter(top1percent.longitude, top1percent.latitude, marker='o', s = 3, c=top1percent.price)

cbar3 = f.colorbar(pts3, ax=ax3)

cbar3.set_label('price')

ax3.set_title("Airbnb Pricing For Top 0.01 Percent")

'''
# check pricing by borough

sns.set_palette('cubehelix')

plt.figure(figsize=(15,10))

scatter = sns.violinplot(x='neighbourhood_group', y='price', data=outliers_gone)

#scatter2 = sns.violinplot(x='neighbourhood_group', y='price', data=outliers)
#availability distribution by room type

sns.set_palette('Paired')

plt.figure(figsize=(15,10))

scatter = sns.violinplot(x='room_type', y='availability_365', data=geo_df)