# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/scrubbed.csv")

data["latitude"][data["latitude"]=="33q.200088"]="33.200088"

data.head()
lats=np.array(data["latitude"].astype(float))

lons=(np.array(data["longitude "].astype(float)))

plt.figure(figsize=(20,10))

map = Basemap()

map.fillcontinents(color='#FDE3A7',lake_color='lightblue',zorder=0.5)

map.drawmapboundary(fill_color='#C5EFF7')

#map.filloceans(color="lightblue")

map.drawcountries()

map.drawcoastlines()

x, y = map(lons, lats)

map.scatter(x, y,color='#03C9A9',alpha=0.8)



plt.show()
data["country"].value_counts()
countries=np.array(data["country"].value_counts())

ind=np.arange(5)

plt.bar(ind,countries,width=0.35)

plt.xlabel("Countries")

plt.ylabel("No. of UFO sightings")

plt.xticks(ind,["US","CA","GB","AU","DE"])

plt.show()