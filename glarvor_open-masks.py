import xarray as xr

import numpy as np

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
zone = "NW"

fname = "/kaggle/input/meteonet/Masks/Masks/%s_masks.nc" % (zone)
data = xr.open_dataset(fname)
fig, axs = plt.subplots(1,2, figsize=(9,4))



# Plot the land-sea mask

data["lsm"].plot(ax=axs[0])



# Plot the relief mask

vmax = data["p3008"].values.max()

vmin = data["p3008"].values.min()



data["p3008"].plot(ax=axs[1], cmap='terrain', vmin=vmin, vmax=vmax)



plt.show()
data
coord = 'longitude'

data[coord]
data[coord].units
data[coord].values[0:10]
data['lsm']
import cartopy.crs as ccrs

import cartopy.feature as cfeature
# Coordinates of studied area boundaries (in °N and °E)

lllat = 46.25  #lower left latitude

urlat = 51.896  #upper right latitude

lllon = -5.842  #lower left longitude

urlon = 2  #upper right longitude

extent = [lllon, urlon, lllat, urlat]
fig = plt.figure()



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



# Plot the data

data["lsm"].plot()



# Add coastlines and borders

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))



# Adjust the plot to the area we defined 

#/!\# this line causes a bug of the kaggle notebook and clears all the memory. That is why this line is commented and so

# the plot is not completely adjusted to the data

# Show only the area we defined

#ax.set_extent(extent)



plt.show()
fig = plt.figure()



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



# Plot the data

vmax = data["p3008"].values.max()

vmin = data["p3008"].values.min()

data["p3008"].plot(cmap='terrain', vmin=vmin, vmax=vmax)



# Add coastlines and borders

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))



# Adjust the plot to the area we defined 

#/!\# this line causes a bug of the kaggle notebook and clears all the memory. That is why this line is commented and so

# the plot is not completely adjusted to the data

# Show only the area we defined

#ax.set_extent(extent)



plt.show()