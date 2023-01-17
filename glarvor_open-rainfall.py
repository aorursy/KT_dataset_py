import numpy as np

import matplotlib.pyplot as plt

from matplotlib import colors



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
zone = "NW"

year = 2016

month = 5

part_month = 3 # Choice between 1,2,3, as we said earlier each month is divided in 3 parts
directory = '/kaggle/input/meteonet/'

fname = directory + f'{zone}_rainfall_{str(year)}/{zone}_rainfall_{str(year)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'

fname_coords = directory + 'Radar_coords/Radar_coords/'+ f'radar_coords_{zone}.npz'



d = np.load(fname, allow_pickle=True)

data = d['data']

dates = d['dates']

miss_dates = d['miss_dates']



coords = np.load(fname_coords, allow_pickle=True)

#it is about coordinates of the center of pixels 

lat = coords['lats']

lon = coords['lons']
title = "4 examples of rainfall plots"

fig, ax = plt.subplots(2, 2,figsize=(9,9))

fig.suptitle(title, fontsize=16)



# Choose the colormap

cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 

                              'skyblue','olive','mediumseagreen','cyan','lime','yellow',

                              'khaki','burlywood','orange','brown','pink','red','plum'])

bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,75]

norm = colors.BoundaryNorm(bounds, cmap.N)



pl=ax[0,0].pcolormesh(lon, lat, data[0,:,:],cmap=cmap, norm=norm)

ax[0,0].set_ylabel('latitude (degrees_north)')

ax[0,0].set_title(str(dates[0]) + " - "+  zone + " zone")



pl=ax[0,1].pcolormesh(lon, lat, data[1,:,:],cmap=cmap, norm=norm)

ax[0,1].set_title(str(dates[1]) + " - "+  zone + " zone")



pl=ax[1,0].pcolormesh(lon, lat, data[2,:,:],cmap=cmap, norm=norm)

ax[1,0].set_xlabel('longitude (degrees_east)')

ax[1,0].set_ylabel('latitude (degrees_north)')

ax[1,0].set_title(str(dates[2]) + " - "+  zone + " zone")



pl=ax[1,1].pcolormesh(lon, lat, data[3,:,:],cmap=cmap, norm=norm)

ax[1,1].set_xlabel('longitude (degrees_east)')

ax[1,1].set_title(str(dates[3]) + " - "+  zone + " zone")



# Plot the color bar

cbar = fig.colorbar(pl,ax=ax.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 

                orientation= 'vertical').set_label('Rainfall (in 1/100 mm) / -1 : missing values')

plt.show()
data.shape
dates.shape
dates[:5]
miss_dates.shape
lat.shape
lat
import cartopy.crs as ccrs

import cartopy.feature as cfeature
# Coordinates of studied area boundaries (in °N and °E)

lllat = 46.25  #lower left latitude

urlat = 51.896  #upper right latitude

lllon = -5.842  #lower left longitude

urlon = 2  #upper right longitude

extent = [lllon, urlon, lllat, urlat]



fig = plt.figure(figsize=(9,10))



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



# Plot the data

plt.imshow(data[2592+24], interpolation='none', origin='upper',cmap=cmap, norm=norm, extent=extent)



# Add coastlines and borders

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))



# Add the colorbar

plt.colorbar(cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 

             orientation= 'horizontal').set_label('Rainfall (in 1/100 mm) / -1 : missing values')

plt.title("Rainfalls - "+ str(dates[2592+24]) + " - "+  zone + " zone")

plt.show()