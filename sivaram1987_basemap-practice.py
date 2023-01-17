import os
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

%matplotlib inline
m = Basemap()
m.drawcoastlines()
# blue marble from nasa

fig = plt.figure(figsize=(16, 12))

m = Basemap()



m.drawcoastlines()

#m.bluemarble(scale=0.1)

m.bluemarble(scale=0.2)

plt.show()
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon

# are the lat/lon values of the lower left and upper right corners

# of the map.

# resolution = 'c' means use crude resolution coastlines.

fig = plt.figure(figsize=(16, 12))

m = Basemap(projection='mill',llcrnrlat=-90,urcrnrlat=90,\

llcrnrlon=-180,urcrnrlon=180,resolution='c')

m.drawcoastlines()

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()



plt.show()
plt.figure(figsize=(16, 12))

# India

m = Basemap(projection='ortho', resolution=None, lat_0=20, lon_0=78)

#Africa

#m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)

m.bluemarble(scale=0.5);

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

#m.bluemarble(scale=0.5);

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

m.etopo(scale=0.5,alpha=0.5);

plt.show()
plt.figure(figsize=(16, 12))



m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

m.etopo(scale=0.5,alpha=0.5);

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

m.bluemarble(scale=0.5,alpha=0.5);

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()



m.drawcountries()

m.bluemarble(alpha=0.5);

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

m.etopo(alpha=0.5);

plt.show()
plt.figure(figsize=(16, 12))

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

m.etopo();

plt.show()
import numpy as np

from itertools import chain



def draw_map(m, scale=0.2):  # draw a shaded-relief image

    m.shadedrelief(scale=scale)



# lats and longs are returned as a dictionary

lats = m.drawparallels(np.linspace(-90, 90, 13))

lons = m.drawmeridians(np.linspace(-180, 180, 13))



# keys contain the plt.Line2D instances

#lat_lines = chain(*(tup[1][0] for tup in lats.items()))

#lon_lines = chain(*(tup[1][0] for tup in lons.items()))

#all_lines = chain(lat_lines, lon_lines)



# cycle through these lines and set the desired style

#for line in all_lines:

    #line.set(linestyle='-', alpha=0.3, color='w')



#plt.figure(figsize=(16, 12),edgecolor='w')

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

#m = Basemap(projection='robin', lat_0=20, lon_0=78)

#m = Basemap(projection='cyl', lat_0=20, lon_0=78)

#m = Basemap(projection='moll', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

#m.etopo(alpha=0.5);



# draw a shaded-relief image

#m.shadedrelief(scale=0.5)



draw_map(m)

plt.show()
plt.figure(figsize=(16, 12),edgecolor='w')



m = Basemap(projection='cyl', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

#m.etopo(alpha=0.5);

# draw a shaded-relief image

m.shadedrelief(scale=0.5)

draw_map(m)

plt.show()
plt.figure(figsize=(16, 12),edgecolor='w')

m = Basemap(projection='ortho', lat_0=20, lon_0=78)

m.drawcoastlines()

m.drawcountries()

#m.etopo(alpha=0.5);

# draw a shaded-relief image

m.shadedrelief(scale=0.5)

draw_map(m)

plt.show()
from itertools import chain

plt.figure(figsize=(16, 12),edgecolor='w')

m = Basemap(projection='cyl', lat_0=50, lon_0=0)

m.drawcoastlines()

m.drawcountries()

#m.etopo(alpha=0.5);

#draw a shaded-relief image

m.shadedrelief(scale=0.5)

draw_map(m)

plt.show()
from itertools import chain

plt.figure(figsize=(8,6),edgecolor='w')

m = Basemap(projection='cyl', lat_0=50, lon_0=0,lat_1=45,lon_1=255)

m.drawcoastlines()



m.drawcountries()

#m.etopo(alpha=0.5);

# draw a shaded-relief image

m.shadedrelief(scale=0.5)

draw_map(m)

plt.show()
# make sure the value of resolution is a lowercase L,

# for 'low', not a numeral 1

map = Basemap(projection='ortho', lat_0=20, lon_0=78,

resolution='l', area_thresh=1000.0)

map.drawcoastlines()

map.drawcountries()

map.fillcontinents(color='coral')

plt.show()
import numpy as np

map = Basemap(projection='ortho', lat_0=20, lon_0=78,

resolution='l', area_thresh=1000.0)



map.drawmapboundary()

map.drawmeridians(np.arange(0, 360, 30))

map.drawparallels(np.arange(-90, 90, 30))

map.drawcountries()

plt.show()
plt.figure(figsize=(8,6))

map = Basemap(projection='robin', lat_0=20, lon_0=-78,

resolution='l', area_thresh=1000.0)

map.drawmapboundary()

map.drawmeridians(np.arange(0, 360, 30))

map.drawparallels(np.arange(-90, 90, 30))

map.drawcountries()

plt.show()



#Robinson projection instead of a globe:
plt.figure(figsize=(8,6))

m = Basemap(projection='mill',

llcrnrlat = 20,

llcrnrlon = -130,

urcrnrlat = 50,

urcrnrlon = -60,

resolution = 'l')



m.drawcoastlines()

m.drawcountries(linewidth=2)

m.drawstates(color='black')

# 29 north 95 west

# Houston, TX

lat, lon = 29.7604, -95.3698

xpt, ypt = m(lon, lat)

m.plot(xpt, ypt, 'r*', markersize=15)



# Boulder, CO



lat, lon = 40.125, -104.237

xpt, ypt = m(lon, lat)

m.plot(xpt, ypt, 'bo')



land_check = m.is_land(xpt, ypt)



print(land_check)



plt.title('Basemap with Title')

plt.show()
plt.figure(figsize=(8,6))

m = Basemap(projection='mill',

llcrnrlat = 25,

llcrnrlon = -130,

urcrnrlat = 50,

urcrnrlon = -60,

resolution = 'l')



m.drawcoastlines()

m.drawcountries(linewidth=2)

m.drawstates(color='black')
plt.figure(figsize=(8,6))

x1 = -20.

x2 = 40.

y1 = 32.

y2 = 64.

m = Basemap(resolution='l',projection='merc',\

llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,\

urcrnrlon=x2,lat_ts=(x1+x2)/2)



lat = 53.565278



lon = 10.001389



m.drawcoastlines(linewidth=1.0);

m.scatter(lon,lat,300)

x,y = m(lon,lat)



m.drawcoastlines(linewidth=1.0);

m.scatter(x,y,300)

plt.show()
plt.figure(figsize=(8,6))

m.drawcoastlines(linewidth=1)

m.drawcountries(linewidth=1)

m.fillcontinents(color='lightgray',lake_color='aqua',alpha=0.5)

m.drawmeridians(np.arange(0,360,10),labels=[False,True,True,False])

m.drawparallels(np.arange(-90,90,10),labels=[False,True,True,False])

m.drawmapscale(33,35,0,40,1000,barstyle='fancy')

m.scatter(x,y,300,zorder=10)

plt.text(x+200000,y-20000,"Hamburg",color='red',size=14);



plt.show()
plt.figure(figsize=(8,6))

m.drawcoastlines(linewidth=1)

m.drawcountries(linewidth=1)

m.fillcontinents(color='lightgray',lake_color='aqua',alpha=0.5)

m.drawmeridians(np.arange(0,360,10))

m.drawparallels(np.arange(-90,90,10))

m.drawmapscale(33,35,0,40,1000,barstyle='fancy')

m.scatter(x,y,300,zorder=10)

plt.text(x+200000,y-20000,"Hamburg",color='red',size=14);

plt.show()
plt.figure(figsize=(16,12))

m.drawcoastlines(linewidth=1)

m.drawcountries(linewidth=1)

m.fillcontinents(color='lightgray',lake_color='aqua',alpha=0.5)

m.etopo()

m.scatter(x,y,300,zorder=10)

plt.show()
plt.figure(figsize=(16,12))

m.drawcoastlines(linewidth=1)

m.drawcountries(linewidth=1)

m.fillcontinents(color='lightgray',alpha=0.5)

m.bluemarble()

m.scatter(x,y,300,zorder=10)

plt.show()
plt.figure(figsize=(16,12))

m.drawcoastlines(linewidth=1)

m.drawcountries(linewidth=1)

m.fillcontinents(color='lightgray',alpha=0.5)

m.shadedrelief()

m.scatter(x,y,300,zorder=10)

plt.show()