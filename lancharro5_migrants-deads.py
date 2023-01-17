# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import plotly.express as px

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Use right path to the file. This one ('/input/...') is only for open it in Kaggle.com

migrants = pd.read_csv("../input/MissingMigrants-Global-2019-03-29T18-36-07.csv")
migrants['lat'], migrants['lon'] = migrants['Location Coordinates'].str.split(',',1).str
migrants['lon'] = [float(x) for x in migrants['lon']]

migrants['lat'] = [float(x) for x in migrants['lat']]
migrants.columns
migrants_coor = migrants[['Region of Incident','Reported Date','Reported Year','Number Dead', 'lat', 'lon']]

migrants_coor = migrants_coor.fillna(0)
migrants_coor.head()
lats = migrants_coor['lat'][:]

lons = migrants_coor['lon'][:]
deads = migrants_coor[['Region of Incident', 'Number Dead']]
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(llcrnrlat=-60,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

m.drawcountries(linewidth=0.2, color="black")

m.shadedrelief(alpha=0.7)

x, y = m(lons, lats)

s1 = migrants_coor['Number Dead']



l1 = plt.scatter([],[], s=20, color="#e60000")

l2 = plt.scatter([],[], s=100, color="#e60000")

l3 = plt.scatter([],[], s=200, color="#e60000")

l4 = plt.scatter([],[], s=1000, color="#e60000")



labels = ["10", "50", "100", "500"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=4, loc = 9, borderpad = 1.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s1*2, marker="o", alpha=1, c="#e60000")

plt.title('Migration Deads',size=30)
deads.groupby('Region of Incident').sum().sort_values('Number Dead', ascending=False)
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(llcrnrlat=25,urcrnrlat=60,llcrnrlon=-10,urcrnrlon=50,resolution='l')

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

#m.drawcountries(linewidth=1.5, color="black")

m.shadedrelief(alpha=0.7)



x, y = m(lons, lats)

s1 = migrants_coor['Number Dead']



l1 = plt.scatter([],[], s=20, color="#e60000")

l2 = plt.scatter([],[], s=100, color="#e60000")

l3 = plt.scatter([],[], s=200, color="#e60000")

l4 = plt.scatter([],[], s=1000, color="#e60000")



labels = ["10", "50", "100", "500"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=4, loc = 8, borderpad = 1.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s1*2, marker="o", alpha=1, c="#e60000")

plt.title('Migration Deads Mediterranean', size=20)
deads.loc[deads['Region of Incident'].isin(['Mediterranean', 'Europe', 'North of Africa'])].sort_values('Number Dead', ascending=False).head(30)
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(llcrnrlat=30,urcrnrlat=43,llcrnrlon=10,urcrnrlon=20,resolution='l')

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

#m.drawcountries(linewidth=1.5, color="black")

m.shadedrelief(alpha=0.7)

x, y = m(lons, lats)

s1 = migrants_coor['Number Dead']



l1 = plt.scatter([],[], s=20, color="#e60000")

l2 = plt.scatter([],[], s=100, color="#e60000")

l3 = plt.scatter([],[], s=200, color="#e60000")

l4 = plt.scatter([],[], s=1000, color="#e60000")



labels = ["10", "50", "100", "500"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=4, loc = 9, borderpad = 1.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s1*2, marker="o", alpha=1, c="#e60000")

plt.title('Migration Deads Lampedusa', size=20)
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(llcrnrlat=0,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60,resolution='l')

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

m.drawcountries(linewidth=0.5, color="black")

m.shadedrelief(alpha=0.7)



x, y = m(lons, lats)

s1 = migrants_coor['Number Dead']



l1 = plt.scatter([],[], s=20, color="#e60000")

l2 = plt.scatter([],[], s=100, color="#e60000")

l3 = plt.scatter([],[], s=200, color="#e60000")

l4 = plt.scatter([],[], s=1000, color="#e60000")



labels = ["10", "50", "100", "500"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=4, loc = 8, borderpad = 1.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s1*2, marker="o", alpha=1, c="#e60000")

plt.title('Migration Deads Center and North America', size=20)
deads.loc[deads['Region of Incident'].isin(['Caribbean', 'Central America', 'US-Mexico Border', 'North America', 'South America'])].sort_values('Number Dead', ascending=False).head(20)
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(llcrnrlat=-40,urcrnrlat=40,llcrnrlon=-30,urcrnrlon=55,resolution='l')

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

m.drawcountries(linewidth=0.5, color="black")

m.shadedrelief(alpha=0.7)



x, y = m(lons, lats)

s1 = migrants_coor['Number Dead']



l1 = plt.scatter([],[], s=20, color="#e60000")

l2 = plt.scatter([],[], s=100, color="#e60000")

l3 = plt.scatter([],[], s=200, color="#e60000")

l4 = plt.scatter([],[], s=1000, color="#e60000")



labels = ["10", "50", "100", "500"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=4, loc = 8, borderpad = 1.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s1*2, marker="o", alpha=1, c="#e60000")

plt.title('Migration Deads Africa', size=20)
deads.loc[deads['Region of Incident'].isin(['Horn of Africa', 'Middle East', 'North Africa', 'Sub-Saharan Africa'])].sort_values('Number Dead', ascending=False).head(30)
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(llcrnrlat=-10,urcrnrlat=60,llcrnrlon=65,urcrnrlon=140,resolution='l')

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

#m.drawcountries(linewidth=1.5, color="black")

m.shadedrelief(alpha=0.7)



x, y = m(lons, lats)

s1 = migrants_coor['Number Dead']



l1 = plt.scatter([],[], s=20, color="#e60000")

l2 = plt.scatter([],[], s=100, color="#e60000")

l3 = plt.scatter([],[], s=200, color="#e60000")

l4 = plt.scatter([],[], s=1000, color="#e60000")



labels = ["10", "50", "100", "500"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=4, loc = 8, borderpad = 1.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s1*2, marker="o", alpha=1, c="#e60000")

plt.title('Migration Deads Asia', size=20)
deads.loc[deads['Region of Incident'].isin(['Central Asia', 'East Asia', 'South Asia', 'Southeast Asia'])].sort_values('Number Dead', ascending=False).head(20)
m_sum = migrants_coor.groupby('Region of Incident').sum()

m_mean = migrants_coor.groupby('Region of Incident').mean()
m_sum['lat_mean'] = m_mean['lat']

m_sum['lon_mean'] = m_mean['lon']

m_sum = m_sum.drop(['lat','lon'],axis=1)
m_sum.columns
lats_2 = m_sum['lat_mean'][:]

lons_2 = m_sum['lon_mean'][:]
plt.figure(figsize=(40,15))

# A basic map

m=Basemap(lat_0=0, lon_0=0)

#m.drawmapboundary(fill_color='white', linewidth=0)

#m.fillcontinents(color='black', alpha=0.3, lake_color='white')

#m.drawcoastlines(linewidth=0.5, color="black")

#m.drawcountries(linewidth=1.5, color="black")

m.shadedrelief(alpha=0.7)



x, y = m(lons_2, lats_2)

s = m_sum['Number Dead']



l1 = plt.scatter([],[], s=50, color="#e60000")

l2 = plt.scatter([],[], s=500, color="#e60000")

l3 = plt.scatter([],[], s=2000, color="#e60000")

l4 = plt.scatter([],[], s=5000, color="#e60000")



labels = ["50", "500", "2000", "5000"]



leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=14,

handlelength=5, loc = 9, borderpad = 3.5,

handletextpad=1, scatterpoints = 1)



m.scatter(x, y, latlon=True, s=s, marker="o", alpha=1, c="#e60000")
m_sum.sort_values('Number Dead', ascending=False)
migrants.groupby('Reported Year').sum()