import numpy as np

import pandas as pd

from datetime import datetime

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



pmd = pd.read_csv('../input/300k.csv',low_memory=False)



lats = pmd.latitude.values

lons = pmd.latitude.values

times = pmd.appearedLocalTime.as_matrix()

time = np.array([datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') for d in times])
pmd.head(10)
# Let's see where the pokemon appeared!

fig = plt.figure(figsize=(10,5))



# Create a map, using the Gall–Peters projection, 

m = Basemap(projection='merc',

             llcrnrlat=-60,

             urcrnrlat=65,

             llcrnrlon=-180,

             urcrnrlon=180,

             lat_ts=0,

             resolution='c')



# Draw the coastlines on the map

m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color = '#888888')

m.drawmapboundary(fill_color='#f4f4f4')

x, y = m(pmd.longitude.tolist(),pmd.latitude.tolist())

m.scatter(x,y, s=3, c="#1292db", lw=0, alpha=1, zorder=5)



# Show the map

plt.title("Pokemon WHOA!")

plt.show()