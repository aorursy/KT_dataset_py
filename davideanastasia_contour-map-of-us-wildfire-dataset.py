import sqlite3

import numpy as np

import pandas as pd
# read lat/lng data into panda dataframe

conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')



df = pd.read_sql_query("SELECT latitude, longitude FROM fires;", conn)



lats = df['LATITUDE']

lons = df['LONGITUDE']
# bounding box of united states

# bbox_ll = [24.356308, -124.848974]

# bbox_ur = [49.384358, -66.885444] 



bbox_ll = [24.0, -125.0]

bbox_ur = [50.0, -66.0] 



# geographical center of united states

lat_0 = 39.833333

lon_0 = -98.583333



# compute appropriate bins to aggregate data

# nx is number of bins in x-axis, i.e. longitude

# ny is number of bins in y-axis, i.e. latitude

nx = 80

ny = 40



# form the bins

lon_bins = np.linspace(bbox_ll[1], bbox_ur[1], nx)

lat_bins = np.linspace(bbox_ll[0], bbox_ur[0], ny)



# aggregate the number of fires in each bin, we will only use the density

density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])



# get the mesh for the lat and lon

lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)
# # Here adding one row and column at the end of the matrix, so that 

# # density has same dimension as lat_bins_2d, lon_bins_2d, otherwise, 

# # using shading='gouraud' will raise error

density = np.hstack((density,np.zeros((density.shape[0],1))))

density = np.vstack((density,np.zeros((density.shape[1]))))
%matplotlib inline  



# plotting map based on example at https://matplotlib.org/basemap/users/examples.html

# install basemap on Mac: sudo -H pip2 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz

from mpl_toolkits.basemap import Basemap, cm

from matplotlib.colors import LinearSegmentedColormap



import matplotlib.pyplot as plt





fig = plt.figure(figsize=(12,8))



ax = fig.add_axes([0.1,0.1,0.8,0.8])



# create polar stereographic Basemap instance.

m = Basemap(projection='merc',

            lon_0=lon_0,lat_0=90.,lat_ts=lat_0,

            llcrnrlat=bbox_ll[0],urcrnrlat=bbox_ur[0],

            llcrnrlon=bbox_ll[1],urcrnrlon=bbox_ur[1],

            rsphere=6371200.,resolution='l',area_thresh=10000)



# draw coastlines, state and country boundaries, edge of map.

m.drawcoastlines()

m.drawstates()

m.drawcountries()



# draw parallels.

parallels = np.arange(0.,90,10.)

m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)



# draw meridians

meridians = np.arange(180.,360.,10.)

m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)



# convert the bin mesh to map coordinates:

# xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh



# create contour lines

CS1 = m.contour(lon_bins_2d, lat_bins_2d, density,15,linewidths=0.5,colors='k',latlon=True)

# fill between contour lines. plt.cm.jet

CS2 = m.contourf(lon_bins_2d, lat_bins_2d, density, CS1.levels, cmap=plt.cm.PuRd, extend='both', latlon=True)

# draw colorbar

m.colorbar(CS2) 



plt.show()