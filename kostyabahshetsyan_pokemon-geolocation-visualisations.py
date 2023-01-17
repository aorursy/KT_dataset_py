# Inspired by 

# https://www.kaggle.com/maddarwin/talkingdata-mobile-user-demographics/one-day-in-china-geolocation-animation/run/306864/code

# and

# https://www.kaggle.com/beyondbeneath/talkingdata-mobile-user-demographics/geolocation-visualisations
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
time_groups = pcm.groupby('appearedHour')

plt.figure(1, figsize=(20,10))

m = map = Basemap(

    projection='merc', 

    llcrnrlat=-60,

             urcrnrlat=65,

             llcrnrlon=-180,

             urcrnrlon=180,

             lat_ts=0,

    resolution='i')

m.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes

m.drawmapboundary(fill_color='#000000')                # black background

m.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders



x,y = m(0, 0)

point = m.plot(x, y, 'o', markersize=2, color='b')[0]

def init():

    point.set_data([], [])

    return point,



def animate(i):

    lon = time_groups.get_group(i)['longitude'].values

    lat = time_groups.get_group(i)['latitude'].values

    x, y = m(lon ,lat)

    point.set_data(x,y)

    plt.title('Pocemon activity at %2d:00' % (i))

    return point,



output = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=24, interval=500, blit=True, repeat=False)

output.save("pocemon.gif", writer='imagemagick')

plt.show()