import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pylab as pl

import matplotlib.gridspec as gridspec

from matplotlib import colors



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
zone = "NW"    # NW or SE

year = 2016    # 2016, 2017 or 2018

month = 5

part_month = 3 # Choice between 1,2,3, as we said earlier each month is divided in 3 parts

#min and max indexes when the dataset is loaded in memory

ind_min = 2592

ind_max = 2692
directory = '/kaggle/input/meteonet/'

fname = directory + f'{zone}_reflectivity_old_product_{str(year)}/{zone}_reflectivity_old_product_{str(year)}/reflectivity-old-{zone}-{str(year)}-{str(month).zfill(2)}/reflectivity-old-{zone}-{str(year)}-{str(month).zfill(2)}/reflectivity_old_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'

fname_coords = directory + 'Radar_coords/Radar_coords/'+ f'radar_coords_{zone}.npz'



d = np.load(fname, allow_pickle=True)

data = d['data'][ind_min:ind_max,:,:]     #reflectivity values

dates = d['dates'][ind_min:ind_max]        #associated dates values

miss_dates = d['miss_dates'][ind_min:ind_max]        #missing dates values



#get the coordinates of the points

coords = np.load(fname_coords, allow_pickle=True)

#it is about coordinates of the center of pixels -> it is necessary to get the coordinates of the center of pixels

lat = coords['lats']

lon = coords['lons']
fig = plt.figure(figsize=(9,9))

gs = gridspec.GridSpec(2, 2, figure = fig)

fig.suptitle("4 examples of reflectivity plots" + ' / ' + str(255) +' : missing values', fontsize=12)



# Reflectivity : colorbar definition

if (np.max(data) > 56):

    borne_max = np.max(data)

else:

    borne_max = 56 + 10

cmap = colors.ListedColormap(['lavender','indigo','mediumblue','dodgerblue', 'skyblue','cyan',

                          'olivedrab','lime','greenyellow','orange','red','magenta','pink','silver'])

bounds = [0,4,8,12,16,20,24,32,40,48,56,borne_max,255]

norm = colors.BoundaryNorm(bounds, cmap.N)



# 4 plot examples

ax =  pl.subplot(gs[0, 0])  

p1=ax.pcolormesh(lon, lat, data[0,:,:],cmap=cmap, norm=norm)

ax.set_ylabel('latitude (degrees_north)')

ax.set_title(str(dates[0]) + " - "+  zone + " zone")

pl.colorbar(p1,ax=ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 

            orientation= 'vertical').set_label('Reflectivity (in dBZ)')



ax =  pl.subplot(gs[0, 1])  

p2=ax.pcolormesh(lon, lat, data[1,:,:],cmap=cmap, norm=norm)

ax.set_title(str(dates[1]) + " - "+  zone + " zone")

pl.colorbar(p2,ax=ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 

            orientation= 'vertical').set_label('Reflectivity (in dBZ)')



ax =  pl.subplot(gs[1, 0])  

p3=ax.pcolormesh(lon, lat, data[2,:,:],cmap=cmap, norm=norm)

ax.set_xlabel('longitude (degrees_east)')

ax.set_ylabel('latitude (degrees_north)')

ax.set_title(str(dates[2]) + " - "+  zone + " zone")

pl.colorbar(p3,ax=ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 

            orientation= 'vertical').set_label('Reflectivity (in dBZ)')



ax =  pl.subplot(gs[1, 1])  

p4=ax.pcolormesh(lon, lat, data[3,:,:],cmap=cmap, norm=norm)

ax.set_xlabel('longitude (degrees_east)')

ax.set_title(str(dates[3]) + " - "+  zone + " zone")

pl.colorbar(p4,ax=ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, 

            orientation= 'vertical').set_label('Reflectivity (in dBZ)')



plt.show()
data.shape
dates.shape
dates[0:5]
miss_dates.shape
miss_dates
lat.shape
lat
import cartopy.crs as ccrs

import cartopy.feature as cfeature
#index of the chosen 5 min of the decade for the plot with cartopy (cf last cell of the notebook) 

ind = 24
# Coordinates of studied area boundaries (in °N and °E)

lllat = 46.25  #lower left latitude

urlat = 51.896  #upper right latitude

lllon = -5.842  #lower left longitude

urlon = 2  #upper right longitude

extent = [lllon, urlon, lllat, urlat]



fig = plt.figure(figsize=(9,10))



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



#colorbar definition

if (np.max(data) > 56):

    borne_max = np.max(data)

else:

    borne_max = 56 + 10

cmap = colors.ListedColormap(['lavender','indigo','mediumblue','dodgerblue', 'skyblue','cyan',

                          'olivedrab','lime','greenyellow','orange','red','magenta','pink','silver'])

bounds = [0,4,8,12,16,20,24,32,40,48,56,borne_max,255]

norm = colors.BoundaryNorm(bounds, cmap.N)



#plot the data and the background map (coastlines and borders)

img = ax.imshow(data[ind,:,:], interpolation='none', origin='upper',cmap=cmap, norm=norm, extent=extent)

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))



plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation= 'horizontal').set_label('Reflectivity (in dBZ) / '+ 

                                                                                                              str(255) +' : missing values')

plt.title("Reflectivity - "+ str(dates[ind]) + " - "+  zone + " zone")

plt.show()
#multiplication coefficient for the graphic representation of rainfall rate (in 1/coeff mm/h)

#/!\### the colorbar is adapted to the coefficient 10 (cf plot at the end)

coeff = 10 
##from dBZ to mm/h : Marshall-Palmer formula

a = 200

b = 1.6



rr = np.zeros((data.shape[0],data.shape[1],data.shape[2]))

rr[data==255]=np.nan

rr[data!=255] = (10**(data[data!=255]/10)/a)**(1/b)



rr_plot = coeff*rr
def plot_mmh(data,coeff, lat,lon, title = "4 examples of rainfall rate plots"):

           

    #plots

    fig, ax = plt.subplots(2, 2,figsize=(9,9))

    fig.suptitle(title, fontsize=16)

    

    #colorbar definition

    #/!\### the scale is adapted to the coefficient 10

    if (np.nanmax(rr_plot) > 3646):

        borne_max = np.nanmax(rr_plot)

    else:

        borne_max = 3646 + 10

    cmap = colors.ListedColormap(['white', 'darkslateblue', 'mediumblue','dodgerblue', 'skyblue','olive','mediumseagreen'

                                      ,'cyan','lime','yellow','khaki','burlywood','orange','brown','red'])

    bounds = [0,2,4,6,12,21,36,65,115,205,365,648,1153,2050,3646,borne_max]

    norm = colors.BoundaryNorm(bounds, cmap.N)

    

    pl=ax[0,0].pcolormesh(lon, lat, rr_plot[0,:,:],cmap=cmap, norm=norm)

    ax[0,0].set_ylabel('latitude (degrees_north)')

    ax[0,0].set_title(str(dates[0]) + " - "+  zone + " zone")



    pl=ax[0,1].pcolormesh(lon, lat, rr_plot[1,:,:],cmap=cmap, norm=norm)

    ax[0,1].set_title(str(dates[1]) + " - "+  zone + " zone")



    pl=ax[1,0].pcolormesh(lon, lat, rr_plot[2,:,:],cmap=cmap, norm=norm)

    ax[1,0].set_xlabel('longitude (degrees_east)')

    ax[1,0].set_ylabel('latitude (degrees_north)')

    ax[1,0].set_title(str(dates[2]) + " - "+  zone + " zone")



    pl=ax[1,1].pcolormesh(lon, lat, rr_plot[3,:,:],cmap=cmap, norm=norm)

    ax[1,1].set_xlabel('longitude (degrees_east)')

    ax[1,1].set_title(str(dates[3]) + " - "+  zone + " zone")



    cbar = fig.colorbar(pl,ax=ax.ravel().tolist(),cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation= 'vertical').set_label('Rainfall rate (in 1/10 mm/h) / nan : missing values')

    plt.show()

    return rr
#/!\ these function works only if there are at least 3 dates later than this chosen for the plot with Cartopy (ind variable, cf second cell)

rr = plot_mmh(data,coeff, lat,lon)
data_plot = coeff*rr



fig = plt.figure(figsize=(9,10))



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



#colorbar definition

#/!\### the scale is adapted to the coefficient 10

if (np.nanmax(data_plot) > 3646):

    borne_max = np.nanmax(data_plot)

else:

    borne_max = 3646 + 10

cmap = colors.ListedColormap(['white', 'darkslateblue', 'mediumblue','dodgerblue', 'skyblue','olive','mediumseagreen'

                                  ,'cyan','lime','yellow','khaki','burlywood','orange','brown','red'])

bounds = [0,2,4,6,12,21,36,65,115,205,365,648,1153,2050,3646,borne_max]

norm = colors.BoundaryNorm(bounds, cmap.N)



#plot the data and the background map (coastlines and borders)

img = ax.imshow(data_plot[ind,:,:], interpolation='none', origin='upper',cmap=cmap, norm=norm, extent=extent)

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))



plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation= 'horizontal').set_label('Rainfall rate (in 1/10 mm/h) / nan : missing values')

plt.title("Rainfall rate - "+ str(dates[ind]) + " - "+  zone + " zone")

plt.show()