
import pandas as pd
import math
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
plotData=train[['DayOfWeek','Category']].groupby(['DayOfWeek','Category']).size()
plotDataDF = pd.DataFrame({'count' : train[['DayOfWeek','Category']].groupby(['DayOfWeek','Category']).size()}).reset_index()

# We will take the example of just the WARRANTS
train[train['Category']=='WARRANTS'].DayOfWeek.value_counts().plot(kind='bar')
plt.xlabel('Day Of Week', fontsize=18)
plt.ylabel('Crime Count', fontsize=16)
plt.show()

# We see that there is a need for including the DayOfWeek

train[train['Category']=='BURGLARY'].DayOfWeek.value_counts().plot(kind='bar')
plt.xlabel('Day Of Week', fontsize=18)
plt.ylabel('Crime Count', fontsize=16)
plt.show()
train[train['Category']=='ASSAULT'].DayOfWeek.value_counts().plot(kind='bar')
plt.xlabel('Day Of Week', fontsize=18)
plt.ylabel('Crime Count', fontsize=16)
plt.show()
train[train['Category']=='DRUNKENNESS'].DayOfWeek.value_counts().plot(kind='bar')
plt.xlabel('Day Of Week', fontsize=18)
plt.ylabel('Crime Count', fontsize=16)
plt.show()

# So we can see that there is a huge impact because of the Day Of the Week
# Let us do similar analysis for the other variables
# Let us analyse the X coordinate

train['XAbs']=[ math.floor(x) for x in train['X']]
train['YAbs']=[ math.floor(x) for x in train['Y']]


train[train['XAbs']==-123].DayOfWeek.value_counts().plot(kind='bar')
plt.xlabel('Day Of Week', fontsize=18)
plt.ylabel('X Coordinate - 123', fontsize=16)
# We will now be drawing the map
# We will now be drawing the map
from mpl_toolkits.basemap import Basemap

map = Basemap(projection='merc', lat_0 = 37.6, lon_0 = -122.5,
    resolution = 'h', area_thresh = 0.1,
    llcrnrlon=-122.55, llcrnrlat=37.7,
    urcrnrlon=-122.35, urcrnrlat=37.9)
 
map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'coral')
map.drawmapboundary()


#lons = [-122.45, -122.57, -122.52]
#lons = train['X'].head(n=5).tolist()
lons = train['X'].tolist()
#lats = [37.81,37.81,37.81]
#lats = train['Y'].head(n=5).tolist()
lats = train['Y'].tolist()
x,y = map(lons, lats)
map.plot(x, y, 'bo', markersize=5)

plt.show()
