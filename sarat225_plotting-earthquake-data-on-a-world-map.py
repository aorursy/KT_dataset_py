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
from pandas import DataFrame

from pandas import Series

import matplotlib.pyplot as plt

earthquake_data = pd.read_csv("../input/database.csv")

earthquake_data.shape
latitude_list = []

longitude_list= []



for row in earthquake_data.Latitude:

    latitude_list.append(row)

for row in earthquake_data.Longitude:

    longitude_list.append(row)
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
%matplotlib inline
earthquake_map = Basemap(projection='robin', lat_0=-90, lon_0=130,resolution='c', area_thresh=1000.0)
earthquake_map.drawcoastlines()

earthquake_map.drawcountries()

earthquake_map.drawmapboundary()

earthquake_map.bluemarble()

earthquake_map.drawstates()

earthquake_map.drawmeridians(np.arange(0, 360, 30))

earthquake_map.drawparallels(np.arange(-90, 90, 30))



x,y = earthquake_map(longitude_list, latitude_list)

earthquake_map.plot(x, y, 'ro', markersize=1)

plt.title("Locations where EarthQuakes,Rock Bursts & NuclearExplosions happened between 1965 to 2016")

 

plt.show()