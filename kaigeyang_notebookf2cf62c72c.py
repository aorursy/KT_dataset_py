# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

from datetime import date

import seaborn as sns

color=sns.color_palette()

train_df=pd.read_json('../input/train.json')

train=train_df.copy()

target=['interest_level']

target=train[target]

train['created']=pd.to_datetime(train.created)

train.head()

llimit=np.percentile(train.latitude.values,1)

ulimit=np.percentile(train.latitude.values,99)

train.latitude.ix[train.latitude<llimit]=llimit

train.latitude.ix[train.latitude>ulimit]=ulimit



llimit=np.percentile(train.longitude.values,1)

ulimit=np.percentile(train.longitude.values,99)

train.longitude.ix[train.longitude<llimit]=llimit

train.longitude.ix[train.longitude>ulimit]=ulimit



llimit=np.percentile(train.price.values,5)

ulimit=np.percentile(train.price.values,95)

train.price.ix[train.price<llimit]=llimit

train.price.ix[train.price>ulimit]=ulimit



train['level']=0

train['level'][train.interest_level=='high']=30

train['level'][train.interest_level=='low']=10

train['level'][train.interest_level=='medium']=20

train.columns
import matplotlib.pyplot as plt

import matplotlib.cm



from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize

west, south, east, north = -74.02, 40.64, -73.85, 40.86

fig=plt.figure(figsize=(15,15))

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='l')

m.drawcoastlines()

m.drawcountries()

# draw parallels and meridians.

parallels = np.arange(-90.,91.,5.)

# Label the meridians and parallels

m.drawparallels(parallels,labels=[True,False,False,False])

# Draw Meridians and Labels

meridians = np.arange(-180.,181.,10.)

m.drawmeridians(meridians,labels=[True,False,False,True])

m.drawmapboundary(fill_color='white')

# Define a colormap

jet = plt.cm.get_cmap('jet')

# Transform points into Map's projection

x,y = m(train.longitude.values,train.latitude.values)

# Color the transformed points!

sc = plt.scatter(x,y, c=train.level.values, vmin=0, vmax =35, cmap=jet, s=20, edgecolors='none')

# And let's include that colorbar

cbar = plt.colorbar(sc, shrink = .5)

cbar.set_label('Interest Level')

plt.title('Scatter of Interest level',fontsize=20)

plt.show()
west, south, east, north = -74.02, 40.64, -73.85, 40.86

fig=plt.figure(figsize=(15,15))

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

m.drawcoastlines()

m.drawcountries()

# draw parallels and meridians.

parallels = np.arange(-90.,91.,5.)

# Label the meridians and parallels

m.drawparallels(parallels,labels=[True,False,False,False])

# Draw Meridians and Labels

meridians = np.arange(-180.,181.,10.)

m.drawmeridians(meridians,labels=[True,False,False,True])

m.drawmapboundary(fill_color='white')

# Define a colormap

jet = plt.cm.get_cmap('jet')

# Transform points into Map's projection

x,y = m(train.longitude.values,train.latitude.values)

# Color the transformed points!

sc = plt.scatter(x,y, c=train.price.values, vmin=min(train.price.values), 

				vmax=max(train.price.values), cmap=jet, s=20, edgecolors='none')

# And let's include that colorbar

cbar = plt.colorbar(sc, shrink = .5)

cbar.set_label('Price')

plt.title('Scatter of Price',fontsize=20)



plt.show()