import numpy as np

import pandas as pd

from mpl_toolkits.basemap import Basemap

from matplotlib import animation

import matplotlib.pyplot as plt

%matplotlib inline
pcm = pd.read_csv('../input/300k.csv', low_memory=False)
pcm[['city','latitude', 'longitude', 'appearedLocalTime']].head(10)
plt.figure(1, figsize=(20,10))

m1 = Basemap(projection='merc',

             llcrnrlat=-60,

             urcrnrlat=65,

             llcrnrlon=-180,

             urcrnrlon=180,

             lat_ts=0,

             resolution='c')



m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes

m1.drawmapboundary(fill_color='#000000')                # black background

m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders



# Plot the data

x, y = m1(pcm.longitude.tolist(),pcm.latitude.tolist())

m1.scatter(x,y, s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.title("Pocemon activity")

plt.show()