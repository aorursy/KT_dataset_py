import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.cm



from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize



import matplotlib.pyplot as plt

import matplotlib.cm

 

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize

df = pd.read_csv('../input/ukTrafficAADF.csv')
fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='i', # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing

            projection='tmerc', # The projection style is what gives us a 2D view of the world for this

            lon_0=-4.36,lat_0=54.7, # Setting the central point of the image

            epsg=27700) # Setting the coordinate system we're using



m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like

m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like

m.drawcoastlines()

m.drawrivers() # Default colour is black but it can be customised

m.drawcountries()



df['lat_lon'] = list(zip(df.Easting, df.Northing)) # Creating tuples

df_2000 = df[df['AADFYear']==2000]



for i in df_2000[0:500]['lat_lon']:

    x,y = i

    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)



plt.show()
fig, ax = plt.subplots(figsize=(10,10))



m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='f',

            projection='cass',

            lon_0=-4.36,lat_0=54.7,

            epsg=27700)



m.bluemarble() # The key change in this cell



m.drawcoastlines() # You can run without this and it removes the black line



df['lat_lon'] = list(zip(df.Easting, df.Northing))

df_2000 = df[df['AADFYear']==2000]



for i in df_2000[0:1000]['lat_lon']:

    x,y = i

    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)



plt.show()
fig, ax = plt.subplots(figsize=(10,10))



m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='f',

            projection='cass',

            lon_0=-4.36,lat_0=54.7,

            epsg=27700)



m.shadedrelief()



m.drawcoastlines() # You can run without this and it removes the black line



df['lat_lon'] = list(zip(df.Easting, df.Northing))

df_2000 = df[df['AADFYear']==2000]



for i in df_2000[0:1000]['lat_lon']:

    x,y = i

    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)



plt.show()
fig, ax = plt.subplots(figsize=(10,10))



m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='f',

            projection='cass',

            lon_0=-4.36,lat_0=54.7,

            epsg=27700)



m.etopo()



m.drawcoastlines() # You can run without this and it removes the black line



df['lat_lon'] = list(zip(df.Easting, df.Northing))

df_2000 = df[df['AADFYear']==2000]



for i in df_2000[0:1000]['lat_lon']:

    x,y = i

    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)



plt.show()
fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='i',

            projection='tmerc',

            lon_0=-4.36,lat_0=54.7,

            epsg=27700)



m.drawmapboundary(fill_color='#e2e2d7')

m.fillcontinents(color='#007fff',lake_color='#ffffff') #zorder=0

#m.drawcoastlines()

m.drawrivers(color = '#ffffff', linewidth=2)

m.drawcountries(linewidth=2)



df['lat_lon'] = list(zip(df.Easting, df.Northing))

df_2000 = df[df['AADFYear']==2000]



region_set = set(df['Region']) # Creates a list of all the unique regions

colour_set = ['#f9ebea','#d5d8dc','#c39bd3','#BA4A00','#17A589','#1E8449','#e2df5d','#2E4053','#F1c40F','#A9DFBF','#F0B27A'] # A list of random colour codes

region_colour_dict = dict(zip(region_set, colour_set)) # Creates a dictionary so each region has a colour code



df_2000 = df_2000.reset_index()



for i, r in df_2000.iterrows(): # Runs over rows returning each one as a series, so we can still use the values to se colour

    x,y = r['lat_lon']

    m.plot(x, y, marker = 'o', c=region_colour_dict[r['Region']], markersize=1, alpha=0.8, latlon=False)

# The main change in this block is using the region dictionary to change the colour code for each marker



plt.show()
print(region_set)
fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='i',

            projection='tmerc',

            lon_0=-4.36,lat_0=54.7,

            epsg=27700)



m.drawmapboundary(fill_color='#232b2b') # Make your map into any style you like #46bcec

m.fillcontinents(color='#A9A9A9',lake_color='#46bcec') # Make your map into any style you like

m.drawcoastlines()

m.drawrivers(linewidth=2, color='#46BCEC') # Default colour is black but it can be customised

m.drawcountries(linewidth=2, color='#ffffff')



df['lat_lon'] = list(zip(df.Easting, df.Northing))

df_2000 = df[df['AADFYear']==2000]



region_set = set(df['Region']) # Creates a list of all the unique regions

colour_set = ['#f9ebea','#d5d8dc','#c39bd3','#BA4A00','#17A589','#1E8449','#e2df5d','#2E4053','#F1c40F','#A9DFBF','#F0B27A'] # A list of random colour codes

region_colour_dict = dict(zip(region_set, colour_set)) # Creates a dictionary so each region has a colour code



df_2000 = df_2000.reset_index()



min_PedalCycles = min(df_2000['PedalCycles'])

max_PedalCycles = max(df_2000['PedalCycles'])

denom = max_PedalCycles - min_PedalCycles



for i, r in df_2000.iterrows(): # Runs over rows returning each one as a series, so we can still use the values to se colour

    x,y = r['lat_lon']

    size = (r['PedalCycles']-min_PedalCycles)/denom

    m.plot(x, y, marker = 'o', c=region_colour_dict[r['Region']], markersize=40*size, alpha=0.8, latlon=False)

# The main change in this block is using the region dictionary to change the colour code for each marker



plt.show()
fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='i',

            projection='tmerc',

            lon_0=-4.36,lat_0=54.7,

            epsg=27700)



m.drawmapboundary(fill_color='#46bcec')

m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') #zorder=0

m.drawcoastlines()

m.drawrivers(linewidth=2, color='#46bcec')

m.drawcountries()



#m.drawmapboundary(fill_color='#232b2b') # Make your map into any style you like #46bcec

#m.fillcontinents(color='#A9A9A9',lake_color='#46bcec') # Make your map into any style you like

#m.drawcoastlines()

#m.drawrivers(linewidth=2, color='#46BCEC') # Default colour is black but it can be customised

#m.drawcountries(linewidth=2, color='#ffffff')





df['lat_lon'] = list(zip(df.Easting, df.Northing))

df_2000 = df[df['AADFYear']==2000]



region_set = set(df['Region']) # Creates a list of all the unique regions

colour_set = ['#382E2E','#471F1F','#521414','#8A0F0F','#990000'] # A list of random colour codes

region_colour_dict = dict(zip(region_set, colour_set)) # Creates a dictionary so each region has a colour code



df_2000 = df_2000.reset_index()



min_AllMotorVehicles = min(df_2000['AllMotorVehicles'])

max_AllMotorVehicles = max(df_2000['AllMotorVehicles'])

denom = max_AllMotorVehicles - min_AllMotorVehicles



for i, r in df_2000.iterrows(): # Runs over rows returning each one as a series, so we can still use the values to se colour

    x,y = r['lat_lon']

    size = (r['AllMotorVehicles']-min_AllMotorVehicles)/denom

    if size < 0.1:

        colour_depth = 0

    elif size <0.06:

        colour_depth = 1

    elif size <0.2:

        colour_depth = 2

    elif size <0.3:

        colour_depth = 3

    elif size <1:

        colour_depth = 4

    m.plot(x, y, marker = 'o', c=colour_set[colour_depth], markersize=0.7, alpha=0.8, latlon=False)

# The main change in this block is using the region dictionary to change the colour code for each marker



plt.show()
df_2000['AllMotorVehicles'].plot(kind="box")
import datetime



fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,

            urcrnrlon=2.7800,urcrnrlat=60.840,

            resolution='i', # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing

            projection='tmerc', # The projection style is what gives us a 2D view of the world for this

            lon_0=-4.36,lat_0=54.7, # Setting the central point of the image

            epsg=27700) # Setting the coordinate system we're using



m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like

m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like

m.drawcoastlines()

m.drawrivers() # Default colour is black but it can be customised

m.drawcountries()



m.nightshade(datetime.datetime.now())





df['lat_lon'] = list(zip(df.Easting, df.Northing)) # Creating tuples

df_2000 = df[df['AADFYear']==2000]



for i in df_2000[0:500]['lat_lon']:

    x,y = i

    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)



plt.show()