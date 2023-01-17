import pandas as pd

import numpy as np

import re

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

from pylab import rcParams

#from shapely.wkt import loads

rcParams['figure.figsize'] = 12, 8

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#import data

cities = pd.read_csv("../input/cities.csv")

cities.head()
cities.coords = cities.coords.apply(lambda s: re.sub('[POINT()]', '', s))

cities.coords = cities.coords.apply(lambda s: s.replace(" ",","))

cities['longitude'] = cities.coords.apply(lambda s : s.split(',')[0])

cities['latitude'] = cities.coords.apply(lambda s : s.split(',')[1])

cities = cities.drop(['coords'],axis=1)

cities.longitude = cities.longitude.astype(float)

cities.latitude = cities.latitude.astype(float)

cities.head()
map = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

map.drawcoastlines()

map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])

map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),labels=[0,0,0,1])

map.drawmapboundary(fill_color='aqua')

map.fillcontinents(color='coral',lake_color='aqua')

longitudes = cities["longitude"].tolist()

latitudes = cities["latitude"].tolist()

x,y = map(longitudes,latitudes)

map.scatter(x,y,color='navy',marker=4,zorder=10)

plt.title('Cities location')

plt.show()