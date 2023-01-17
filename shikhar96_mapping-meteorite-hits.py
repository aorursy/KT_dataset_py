import csv

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
filename = '../input/meteorite-landings.csv'

lats, long = [], []

mass = []

with open(filename) as f:

    reader = csv.reader(f)

    next(reader)

    try:

        for row in reader:

            lats.append(float(row[7]))

            long.append(float(row[8]))

        

    except ValueError:

        print("Error in " + str(reader))

        next(reader)
plt.figure(figsize = (16,12))

mt_map = Basemap(projection = 'robin', resolution = 'l', area_thresh = 1000.0, lat_0 = 0,

                lon_0 = -130)

mt_map.drawcoastlines()

mt_map.drawcountries()

mt_map.fillcontinents(color = 'gray')

mt_map.drawmapboundary()

mt_map.drawmeridians(np.arange(0,360,30))

mt_map.drawparallels(np.arange(-90,90,30))

x,y = mt_map(long, lats)

mt_map.plot(x, y, 'ro', 6)

plt.show()
