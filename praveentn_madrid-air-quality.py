# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/stations.csv')
data.head()
data.describe()
data.shape
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline
plt.scatter(data['lon'], data['elevation'])
plt.scatter(data['lat'], data['elevation'])
lats = list(data['lat'])
lons = list(data['lon'])
lats, lons
# How much to zoom from coordinates (in degrees)
zoom_scale = 0

# Setup the bounding box for the zoom and bounds of the map
bbox = [np.min(lats)-zoom_scale,np.max(lats)+zoom_scale,\
        np.min(lons)-zoom_scale,np.max(lons)+zoom_scale]
bbox
# bbox = map(lambda x: x * 1.5, bbox)
# bbox = list(bbox)
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
plt.figure(figsize=(20,13))
# Define the projection, scale, the corners of the map, and the resolution.
m = Basemap(projection='merc',llcrnrlat=bbox[0],urcrnrlat=bbox[1],
            llcrnrlon=bbox[2],urcrnrlon=bbox[3],lat_ts=10,resolution='l')

# Draw coastlines and fill continents and water with color
m.drawcoastlines()
m.fillcontinents(color='peru',lake_color='dodgerblue')

# draw parallels, meridians, and color boundaries
m.drawparallels(np.arange(bbox[0],bbox[1],(bbox[1]-bbox[0])/5),labels=[1,0,0,0])
m.drawmeridians(np.arange(bbox[2],bbox[3],(bbox[3]-bbox[2])/5),labels=[0,0,0,1],rotation=45)
m.drawmapboundary(fill_color='dodgerblue')

# build and plot coordinates onto map
x,y = m(lons,lats)
m.plot(x,y,'r*',markersize=5)
plt.title("Madrid Air Quality Station Distribution")
plt.show()
with pd.HDFStore('../input/madrid.h5') as hdfs:
    df = hdfs['master']
df.shape
df.describe()
with pd.HDFStore('../input/madrid.h5') as hdfs:
    for k in hdfs.keys():
        print('{}: {}'.format(k, ', '.join(hdfs[k].columns)))
with pd.HDFStore('../input/madrid.h5') as hdfs:
    test = hdfs['28079057']
type(test)
test.shape, test.describe()
test.rolling(window=24).mean().plot(figsize=(20, 7), alpha=0.8)
partials = list()

with pd.HDFStore('../input/madrid.h5') as hdfs:
    stations = [k[1:] for k in hdfs.keys() if k != '/master']
    for station in stations:
        df = hdfs[station]
        df['station'] = station
        partials.append(df)
            
df = pd.concat(partials, sort=False).sort_index()

df.info()
df.describe()
df.head()
df.columns
df.index
# df['date'] = df.index.tolist()
df.rolling(window=44).mean().plot(figsize=(20, 7), alpha=0.8)
with pd.HDFStore('../input/madrid.h5') as hdfs:
    for k in hdfs.keys():
        name = k.replace('/','')
        tmp = hdfs[name]
        tmp.rolling(window=44).mean().plot(figsize=(23,11), alpha=0.85)
