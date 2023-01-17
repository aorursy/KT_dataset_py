import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.basemap import Basemap, cm

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))
dataset = pd.read_csv('../input/environmental-remediation-sites.csv')

df = pd.DataFrame(dataset)

df.head()
from matplotlib.colors import LinearSegmentedColormap



m = Basemap(projection='ortho',lon_0=-76.25,lat_0=42.5,resolution='l',\

             llcrnrx=-550*550,llcrnry=-550*550,

             urcrnrx=+600*600,urcrnry=+600*600)



m.drawcoastlines()

m.drawcountries()

m.drawstates()



lats = df['Latitude'].tolist()

lons = df['Longitude'].tolist()



# ######################################################################

# Using the heatmap code from http://bagrow.com/dsv/heatmap_basemap.html

# on this dataset. Credit to James Bagrow, james.bagrow@uvm.edu

#

# ######################################################################

# bin the epicenters (adapted from 

# http://stackoverflow.com/questions/11507575/basemap-and-density-plots)

#

# compute appropriate bins to chop up the data:

db = 1 # bin padding

lon_bins = np.linspace(min(lons)-db, max(lons)+db, 20+1) # 20 bins

lat_bins = np.linspace(min(lats)-db, max(lats)+db, 20+1) # 20 bins

    

density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])



# ######################################################################

# Turn the lon/lat of the bins into 2 dimensional arrays ready

# for conversion into projected coordinates

lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)



# convert the bin mesh to map coordinates:

xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh

# #####################################################################



# define custom colormap, white -> nicered, #E6072A = RGB(0.9,0.03,0.16)

cdict = {'red':  ( (0.0,  1.0,  1.0),

                   (1.0,  0.9,  1.0) ),

         'green':( (0.0,  1.0,  1.0),

                   (1.0,  0.03, 0.0) ),

         'blue': ( (0.0,  1.0,  1.0),

                   (1.0,  0.16, 0.0) ) }

custom_map = LinearSegmentedColormap('custom_map', cdict)

plt.register_cmap(cmap=custom_map)



# add histogram squares and a corresponding colorbar to the map:

plt.pcolormesh(xs, ys, density, cmap="custom_map")



cbar = plt.colorbar(orientation='horizontal', shrink=0.625, aspect=20, fraction=0.2,pad=0.02)

cbar.set_label('Number of brownfield sites',size=18)

#plt.clim([0,100])



# translucent blue scatter plot of epicenters above histogram:    

x,y = m(lons, lats)

m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.33)





plt.gcf().set_size_inches(15,15)





plt.show()