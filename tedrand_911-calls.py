import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.head()
# Create a figure of size (i.e. pretty big)

fig = plt.figure(figsize=(20,10))



# Create a map, using the Gall–Peters projection, 

map = Basemap(width=120000,height=90000,projection='lcc',

              # with low resolution,

              resolution = 'l', 

              # And threshold 100000

              area_thresh = 100000.0,

              # Centered at 0,0 (i.e null island)

              lat_0=40.25,lon_0=-75.5)



# Draw the coastlines on the map

map.drawcoastlines()



# Draw country borders on the map

map.drawcountries()



# Fill the land with grey

# map.fillcontinents(color = '#888888')



# Draw the map boundaries

# map.drawmapboundary(fill_color='#f4f4f4')



# Define our longitude and latitude points

# We have to use .values because of a wierd bug when passing pandas data

# to basemap.

x,y = map(df['lng'].values, df['lat'].values)



# Plot them using round markers of size 6

map.plot(x, y, 'ro', markersize=6)



# Show the map

plt.show()