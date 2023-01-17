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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from geopy.geocoders import GoogleV3

from pygeocoder import Geocoder
accidents = pd.read_csv('../input/accident.csv')
print (accidents.head(10))
print(accidents.tail(10))
## Plotting the states in USA ##
print(accidents.STATE.unique())
print (accidents.info())
# Plotting data on map: using Latitude and Longitude

# Convert longitude and latitude to a location

results = Geocoder.reverse_geocode(accidents['LATITUDE'][0], accidents['LONGITUD'][0])



#lon=[]  

#lat=[]  

#for x in accidents['latitude'][1:1000]: lat.append(x)  

#for x in accidents['longitude'][1:1000]: lon.append(x)
from mpl_toolkits.basemap import Basemap



plt.figure(figsize = (12,6)) # make it bigger first



m = Basemap(projection = 'robin', lon_0 = 45, resolution = 'c') 

#set a background colour 

m.drawmapboundary(fill_color = '#85A6D9')

m.fillcontinents(color = 'white', lake_color = '#85A6D9') 

m.drawcoastlines(color = '#6D5F47', linewidth = .4) 

m.drawcountries(color = '#6D5F47', linewidth = .4)

m.drawmeridians(np.arange(-180, 180, 30), color = '#bbbbbb') 

m.drawparallels(np.arange(-90, 90, 30), color = '#bbbbbb')



x,y = m(lon, lat)

m.plot(x, y, 'ro', markersize=6)



plt.show()