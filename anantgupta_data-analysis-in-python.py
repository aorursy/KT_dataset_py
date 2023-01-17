import numpy as np
import pandas as pd

# Any results you write to the current directory are saved as output.
app_events=pd.read_csv("../input/app_events.csv")
app_labels=pd.read_csv("../input/app_labels.csv")
events=pd.read_csv("../input/events.csv")
label_categories=pd.read_csv("../input/label_categories.csv")
gender_age_train=pd.read_csv("../input/gender_age_train.csv")
phone_brand_device_model=pd.read_csv("../input/phone_brand_device_model.csv")

# Load Libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap

# Just observing the data
#print(app_events.head(5))
#print(app_labels.head(5))
#print(events.head(5))
#print(label_categories.head(5))
#print(phone_brand_device_model.head(5))

# We will try with around 10000 samples
events_subset = events.sample(n=10000)
plt.figure(1, figsize=(12,6))

# Mercator of World
#m1 = Basemap(projection='merc',
#             llcrnrlat=-60,
#             urcrnrlat=65,
#             llcrnrlon=-180,
#             urcrnrlon=180,
#             lat_ts=0,
#             resolution=None)

m1 = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=39.,lat_2=39,lat_0=36,lon_0=104.)
m1.shadedrelief()
#plt.show()

#m1.drawmapboundary(fill_color='#000000')                # black background
#m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders
#m1.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)
#m1.bluemarble()
#m1.fillcontinents(color='w')

# Plot the sample data
#plt.figure()
#m1.shadedrelief()
mxy = m1(events_subset["longitude"].tolist(), events_subset["latitude"].tolist())
#m1.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)
m1.scatter(mxy[0], mxy[1], s=3, c="b", lw=0, alpha=1, zorder=5)
plt.title("Shaded view, to identify the location roughly for city clusters")
plt.show()

# We will now be dividing the X,Y coordinates into grids of 1*1
import math
minX=min(events_subset['latitude'])
maxX=max(events_subset['latitude'])
minY=min(events_subset['longitude'])
maxY=max(events_subset['longitude'])

events_subset['gridX']=events_subset['latitude'].map(lambda x:math.floor(x))
events_subset['gridY']=events_subset['longitude'].map(lambda x:math.floor(x))
eventGroup=events_subset.groupby(['gridX','gridY'])['event_id'].count()
eventGroupdf=pd.DataFrame(eventGroup.values,columns=['Count'])
eventGroupdf['XY']=eventGroup.index.values
eventGroupdf['X']=eventGroupdf.apply(lambda x:x['XY'][0],axis=1)
eventGroupdf['Y']=eventGroupdf.apply(lambda x:x['XY'][1],axis=1)

m1 = Basemap(width=12000000,height=9000000,projection='lcc',
            resolution=None,lat_1=36.,lat_2=36,lat_0=36,lon_0=104.)
m1.shadedrelief()
mxy = m1(eventGroupdf["Y"].tolist(), eventGroupdf["X"].tolist())
m1.scatter(mxy[0], mxy[1], s=eventGroupdf['Count'], c="b", lw=0, alpha=1, zorder=5)
plt.title("Shaded view with COUNT based on lat/long grids")
plt.show()