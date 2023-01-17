import numpy as np 

import csv

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
filename = '../input/ufo_sighting_data.csv'

lats, lons = [], [] 

with open(filename) as f:

    reader = csv.reader(f)

    next(reader)

    for row in reader:

        try:

            lats.append(float(row[9]))

            lons.append(float(row[10]))

        except ValueError:

            next(reader)
plt.figure(figsize=(16,12))   

eq_map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0,

              lat_0=0, lon_0=-130)

eq_map.drawcoastlines()

eq_map.drawcountries()

eq_map.fillcontinents(color = 'gray')

eq_map.drawmapboundary()

eq_map.drawmeridians(np.arange(0, 360, 30))

eq_map.drawparallels(np.arange(-90, 90, 30))

x,y = eq_map(lons, lats)

eq_map.plot(x, y, 'ro', markersize=2)



plt.show()