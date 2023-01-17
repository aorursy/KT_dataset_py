import xarray as xr

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
zone = "NW"     #geographic zone (NW or SE)

model = 'arpege' #weather model (arome or arpege)

MODEL = 'ARPEGE' #weather model (AROME or ARPEGE)

level = '2m'      #vertical level (2m, 10m, P_sea_level or PRECIP)

date = dt.datetime(2016, 2, 14,0,0) # Day example 

#parameter name in the file (cf cells below to know the parameter names -> exploration of metadata)

if level == '2m':

    param = 't2m'

elif level == '10m':

    param = 'u10'

elif level == 'PRECIP':

    param = 'tp'

else:

    param = 'msl'
directory = '/kaggle/input/meteonet/' + zone + '_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'

fname = directory + f'{MODEL}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'

data = xr.open_dataset(fname)  
data.isel(step=[0, 6, 12, 23])[param].plot(x='longitude',

                                           y='latitude',

                                           col='step',

                                           col_wrap=2)
data
coord = 'longitude'

data[coord]
data[coord].units
data[coord].values[0:10]
run_date = data['time']

#run_date.values     #get the values

run_date
range_forecasts_dates = data['valid_time']

range_forecasts_dates
if (level =='2m' or level == '10m'):

    level_name = 'heightAboveGround'

elif (level =='P_sea_level'):

    level_name = 'meanSea'

else:

    level_name = 'surface'

info_level = data[level_name]

info_level
d = data[param]     #param : parameter name defined at the beginning of the Notebook 

d_vals=d.values     #get the values

###examples to get the information from attributes

#d.units                      #unit

#d.long_name                      #long name

d
d_vals.shape
import cartopy.crs as ccrs

import cartopy.feature as cfeature
#index for the studied time step

step = 0                
# Coordinates of studied area boundaries (in °N and °E)

lllat = 46.25  #lower left latitude

urlat = 51.896  #upper right latitude

lllon = -5.842  #lower left longitude

urlon = 2  #upper right longitude

extent = [lllon, urlon, lllat, urlat]



fig=plt.figure(figsize=(9,10))



# Select projection

ax = plt.axes(projection=ccrs.PlateCarree())



#plot the data and the background map (coastlines and borders)

img = ax.imshow(d_vals[step,:,:], interpolation='none', origin='upper', extent=extent)

ax.coastlines(resolution='50m', linewidth=1)

ax.add_feature(cfeature.BORDERS.with_scale('50m'))





plt.colorbar(img, orientation= 'horizontal').set_label(d.long_name+ ' (in '+d.units+ ')')

plt.title(model +" model - "+str(d['valid_time'].values[step])+" - " +zone + " zone")

plt.show()