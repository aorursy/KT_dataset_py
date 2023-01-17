# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import seaborn as sns

%matplotlib inline
infile = "../input/uber-raw-data-sep14.csv"

df = pd.read_csv(infile)

df.head()
print(df.Lat.max(), df.Lat.min())

print(df.Lon.max(), df.Lon.min())
plt.scatter(range(df.shape[0]), np.sort(df.Lat.values))

plt.title("Latitude distribution")

plt.show()
plt.scatter(range(df.shape[0]), np.sort(df.Lon.values))

plt.title("Longitude distribution")

plt.show()
plt.figure(figsize=(8,8))

themap = Basemap(projection='gall',

              llcrnrlon = -74.1,              # lower-left corner longitude

              llcrnrlat = 40.6,               # lower-left corner latitude

              urcrnrlon = -73.8,               # upper-right corner longitude

              urcrnrlat = 40.9,               # upper-right corner latitude

              resolution = 'i',

              area_thresh = 100.0

              )



themap.drawcoastlines()

themap.drawstates()

themap.drawcounties()

themap.drawcountries()

themap.fillcontinents(color = 'gainsboro')

themap.drawmapboundary(fill_color='steelblue')



df['Lon'].ix[df['Lon'] > -73.8] = -73.8

df['Lon'].ix[df['Lon'] < -74.1] = -74.1

df['Lat'].ix[df['Lat'] > 40.9] = 40.9

df['Lat'].ix[df['Lat'] < 40.6] = 40.6



x, y = themap(np.array(df['Lon']), np.array(df['Lat']))

themap.plot(x, y, 

            'o',                    # marker shape

            color='Indigo',         # marker colour

            markersize=0.2            # marker size

            )



plt.show()
df['Lon'].ix[df['Lon'] > -73.8] = -73.8

df['Lon'].ix[df['Lon'] < -74.1] = -74.1

df['Lat'].ix[df['Lat'] > 40.9] = 40.9

df['Lat'].ix[df['Lat'] < 40.6] = 40.6



plt.figure(figsize=(8,8))

sns.set_style("darkgrid", {'axes.grid' : False})

plt.scatter(np.array(df['Lon']), np.array(df['Lat']), s=0.2, alpha=0.6)

plt.xlim([-74.1,-73.8])

plt.ylim([40.6, 40.9])

plt.show()