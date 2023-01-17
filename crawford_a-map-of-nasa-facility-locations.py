import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# This is the mapping library

# https://matplotlib.org/basemap/

from mpl_toolkits.basemap import Basemap
# Import dataset

fnasa='../input/NASA_Facilities.csv'

nasa = pd.read_csv(fnasa)

latlon = nasa.Location
def stripper(thing):

    x,y = thing.split()[1:]

    x = float(x.strip('(,)'))

    y = float(y.strip('(,)'))

    return([x,y])



x = []

y = []



stripped = [stripper(i) for i in latlon]

for i in stripped:

    y.append(i[0])

    x.append(i[1])
plt.figure(figsize=(12, 12))



m = Basemap(width=12000000/2,height=9000000/2,

            projection='lcc',

            resolution='c',

            lat_0=40,

            lon_0=-100)



xpt,ypt = m(x,y)



m.drawcountries()

m.bluemarble()



m.scatter(x, y, latlon=True, marker="o", color="white", zorder=10, edgecolor="black")

plt.show()