# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/earthquake-database/database.csv')
df.head()
vol = pd.read_csv('../input/volcano-eruptions/volcano_data_2010.csv')
print(vol.columns)
vol = vol[['Year', 'Month', 'Day', 'Latitude', 'Longitude', 'Type']].dropna()
vol['Date'] = pd.to_datetime(vol[['Year', 'Month', 'Day']])
vol = vol.drop(columns = {'Year', 'Month', 'Day'})
vol.head()

df.columns
df = df[['Date', 'Latitude', 'Longitude', 'Magnitude', 'Type']]
df.head()
df['Date'] = pd.to_datetime(df['Date'])
print(set(df['Type']))
df.head()
print('Size of the Dataframe', df.shape)
eq = df[df['Type'] == 'Earthquake']
others = df[df['Type'] != 'Earthquake']
# Earthquake, Volcanic eruptions and Nuclear Explosions regions on World Map
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


fig = plt.figure(figsize = (22, 20))
wmap = Basemap()
longitudes = eq['Longitude'].tolist()
latitudes = eq['Latitude'].tolist()
x_eq, y_eq = wmap(longitudes, latitudes)
longitudes = others['Longitude'].tolist()
latitudes = others['Latitude'].tolist()
x_oth, y_oth = wmap(longitudes, latitudes)
longitudes = vol['Longitude'].tolist()
latitudes = vol['Latitude'].tolist()
x_vol, y_vol = wmap(longitudes, latitudes)
plt.title('Earthquake effective Areas')
wmap.drawcoastlines()
wmap.shadedrelief()
wmap.scatter(x_eq, y_eq, s = 5, c = 'r', alpha = 0.2)
wmap.scatter(x_oth, y_oth, s = 10, c = 'g')
wmap.scatter(x_vol, y_vol, s = 10, c = 'b')
# draw parallels
wmap.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])
# draw meridians
wmap.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
ax = plt.gca()
red_patch = mpatches.Patch(color='r', label='Earthquake')
green_patch = mpatches.Patch(color='g', label='Nuclear Explosion/Rockburst/Others')
blue_patch = mpatches.Patch(color='b', label='Volcanic Eruptions')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.legend()
plt.show()
# Effected Areas with Magnitude Heatmap
fig = plt.figure(figsize = (22, 20))
wmap = Basemap()
longitudes = eq['Longitude'].tolist()
latitudes = eq['Latitude'].tolist()
x_eq, y_eq = wmap(longitudes, latitudes)
wmap.drawcoastlines()
wmap.shadedrelief()
# draw parallels
wmap.drawparallels(np.arange(-90,90,20),labels=[1,1,0,1])
# draw meridians
wmap.drawmeridians(np.arange(-180,180,20),labels=[1,1,0,1])
plt.title('Earthquake effective Areas with Magnitude Colormap')
sc =wmap.scatter(x_eq, y_eq, s = 30, c = eq['Magnitude'], vmin=5, vmax =9, cmap='OrRd', edgecolors='none')
cbar = plt.colorbar(sc, shrink = .5)
cbar.set_label('Magnitude')
plt.show()

