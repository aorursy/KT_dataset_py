# Import required pakages



import numpy as np

import pandas as pd 

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

# Load training data

train = pd.read_csv("../input/train.csv")
# Prepare basemap

m = Basemap(projection="mill",llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60,resolution='c')

m.drawcoastlines()

m.drawcountries() 

m.drawstates()

m.fillcontinents(color='#04BAE3',lake_color='#FFFFFF')

m.drawmapboundary(fill_color="#FFFFFF")

parallels = np.arange(0.,81,10.)

meridians = np.arange(0.,360.,10.)



# Pickup coordinates

x,y = m(train['pickup_longitude'].as_matrix(),train['pickup_latitude'].as_matrix())

m.plot(x,y,'ro',alpha=.5)



# Dropoff coordinates

x,y = m(train['dropoff_longitude'].as_matrix(),train['dropoff_latitude'].as_matrix())

m.plot(x,y,'go',alpha=.5)



plt.title("Original Pickups & Dropoffs")

plt.show()
# From the map, we can see some pickup/dropoff locations dot in Ocean or California, where are impossibly reached by Taxi. Those dots should be removed from subsequent analysis.



# Remove unwanted dots.

# Draw two circles with centers (mean(pickup_longitude),mean(pickup_latitude)) and (mean(dropoff_longitude),mean(dropoff_latitude))

# Dots located out of one of the circles are unwanted, and removed from data set.



r = 6

biasx = -2

biasy = 0

center = train.describe().get(['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']).xs('mean')

#get multi row train.describe().get(['pickup_longitude','pickup_latitude']).iloc[[1,2,3]]

train = train[((np.square(train.pickup_longitude-center.get('pickup_longitude')-biasx)+np.square(train.pickup_latitude-center.get('pickup_latitude')-biasy))<=np.square(r)) & ((np.square(train.dropoff_longitude-center.get('dropoff_longitude')-biasx)+np.square(train.dropoff_latitude-center.get('dropoff_latitude')-biasy))<=np.square(r))]



# Prepare basemap

m = Basemap(projection="mill",llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60,resolution='c')

m.drawcoastlines()

m.drawcountries()

m.drawstates()

m.fillcontinents(color='#04BAE3',lake_color='#FFFFFF')

m.drawmapboundary(fill_color="#FFFFFF")

parallels = np.arange(0.,81,10.)

meridians = np.arange(0.,360.,10.)



# Pickup coordinates

x,y = m(train['pickup_longitude'].as_matrix(),train['pickup_latitude'].as_matrix())

m.plot(x,y,'ro',alpha=.5)



# Dropoff coordinates

x,y = m(train['dropoff_longitude'].as_matrix(),train['dropoff_latitude'].as_matrix())

m.plot(x,y,'go',alpha=.5)



plt.title("Validated Pickups & Dropoffs")

plt.show()