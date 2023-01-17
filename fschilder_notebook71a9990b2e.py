from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

data_crimes = pd.read_csv('../input/crimes.csv', header=0)

data_crimes
my_map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,

    resolution = 'h', area_thresh = 0.1,

    llcrnrlon=-94.265011, llcrnrlat=42.977753,

    urcrnrlon=-92.265011, urcrnrlat=44.977753)

 

my_map.drawcoastlines()

my_map.drawcountries()

my_map.fillcontinents(color = 'coral')

my_map.drawmapboundary()

 

lons = np.asarray(data_crimes['Long'])

lats = np.asarray(data_crimes["Lat"])

x,y = my_map(lons, lats)

my_map.plot(x, y, 'bo', markersize=10)

 

labels = np.asarray(data_crimes['Description'])

for label, xpt, ypt in zip(labels, x, y):

    plt.text(xpt, ypt, label)

 

plt.show()
np.asarray(lons).size