import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

%matplotlib inline

sns.set(style="white", context="notebook")



# Load data into pandas dataframe.

df_storelocation = pd.read_csv('../input/directory.csv')
df_storelocation.head()
df_storelocation.tail()
df_storelocation.shape
df_storelocation.notnull().sum()/df_storelocation.shape[0]
print (df_storelocation.shape[0])

print (df_storelocation.Country.value_counts().head())
# Set up the matplotlib figure

f1, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

sns.barplot(df_storelocation.Country.value_counts().head().index, df_storelocation.Country.value_counts().head(), palette="BuGn_d", ax=ax1)

ax1.set_ylabel("Store_Number")
###6. Starbucks In China
df_storelocation[df_storelocation["Country"] == "CN"].head()
# How many stores in China?

df_storelocation[df_storelocation["Country"] == "CN"].shape[0]
df_CN = df_storelocation[df_storelocation["Country"] == "CN"]

# Set up the matplotlib figure

f2, (ax2) = plt.subplots(1, 1, figsize=(8, 6), sharex=True)

sns.barplot(df_CN.City.value_counts().head().index, df_CN.City.value_counts().head(), palette="BuGn_d", ax=ax2)

ax2.set_ylabel("Store_Number")
plt.figure(figsize=(8,8))

map = Basemap(projection='stere', 

              lat_0=35, lon_0=110,

              llcrnrlon=82.33, 

              llcrnrlat=3.01, 

              urcrnrlon=138.16, 

              urcrnrlat=53.123,resolution='l',area_thresh=10000,rsphere=6371200.)

map.drawcoastlines()

map.drawcountries()

map.drawmapboundary(fill_color='white')



# Load in Longitude and Latitude data

df_CN_Longitude = df_CN["Longitude"].astype(float)

df_CN_Latitude = df_CN["Latitude"].astype(float)

x, y = map(list(df_CN_Longitude), list(df_CN_Latitude))

map.plot(x, y, 'bo', markersize = 5)

plt.title('Starbucks In China')

plt.show()