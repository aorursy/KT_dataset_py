import xarray as xr

import datetime as dt



import numpy as np

import pandas as pd

from scipy.interpolate import griddata



import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import matplotlib.pylab as pl

from matplotlib import colors





# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
####Cell containing the modifiable fields######

zone = 'NW'

year = 2016

month = 8

part_month = 3

ind = 15   #index of the chosen 5 min of the decade 

nan_value = -1  #nan value for data (ex : rainfall here)



rain_param = 'rainfall'    #parameter name for rainfall

mask_param = 'lsm'         #parameter name for land-sea mask (cf meta-data in the mask GRIB file)



directory = '/kaggle/input/meteonet/'

rain_fname = directory + f'{zone}_rainfall_{str(year)}/{zone}_rainfall_{str(year)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'

rain_coords_fname = directory + 'Radar_coords/Radar_coords/'+ f'radar_coords_{zone}.npz'



mask_fname = directory + 'Masks/Masks/%s_masks.nc' % (zone)

def radar_to_xarray(rain_fname,rain_coords_fname,ind):

    

    #load data

    d = np.load(rain_fname, allow_pickle=True)

    data = d['data'][ind,:,:]

    

    coords = np.load(rain_coords_fname, allow_pickle=True)

    #it is about coordinates of the center of pixels 

    lat = coords['lats']

    lon = coords['lons']

    

    data = xr.DataArray(data,coords=[lat[:,0],lon[0,:]],dims=['latitude','longitude'])

    d_radar = data.to_dataset(name = 'rainfall')

    

    return d_radar,lat,lon
d_radar,lat,lon = radar_to_xarray(rain_fname,rain_coords_fname,ind)
#ori : for original, data to interpolate

ori_data = d_radar

ori_param = rain_param

#tar : for target, which corresponds to the target grid

tar_fname = mask_fname

tar_param = mask_param

#nan value

nan_value = -1  #nan value for data (ex : rainfall here)
data_to_interpolate = d_radar

target_data = xr.open_dataset(tar_fname) 
# convert missing data (from value to 'nan')

nan_data_to_interpolate = data_to_interpolate.where(data_to_interpolate["rainfall"]!=nan_value)  

#today, with the function above, 2 interpolation methods are implemented for 2D arrays : 'linear' and 'nearest' for nearest neighbors

interpolated_data = nan_data_to_interpolate.interp_like(target_data,method='linear')    
#/!\#### the plots options depend on the GRIB file structure###

fig = plt.figure()

widths = [3.6,9, 3.6]

heights = [9, 3.6]

spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,

                         height_ratios=heights)

ax = fig.add_subplot(spec[0,1])



#colorbar definition for rainfall

if (np.max(data_to_interpolate[ori_param].values) > 65):

    borne_max = np.max(data_to_interpolate[ori_param].values)

else:

    borne_max = 65 + 10

cmap = colors.ListedColormap(['silver','white', 'darkslateblue', 'mediumblue','dodgerblue', 'skyblue','olive','mediumseagreen'

                              ,'cyan','lime','yellow','khaki','burlywood','orange','brown','pink','red','plum'])

bounds = [-1,0,2,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,borne_max]

norm = colors.BoundaryNorm(bounds, cmap.N)



#data to interpolate with nan

fig.subplots_adjust(wspace=0, hspace=0)

label = 'Width: {}\nHeight: {}'.format(widths[1], heights[0])

ax.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')

plt.imshow(nan_data_to_interpolate[ori_param].values,cmap=cmap, norm=norm)

ax.set_title('Rainfall (original grid)')



#interpolated data

ax = fig.add_subplot(spec[1,0])

fig.subplots_adjust(wspace=0, hspace=0)

label = 'Width: {}\nHeight: {}'.format(widths[0], heights[1])

ax.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')

plt.imshow(interpolated_data[ori_param].values,cmap=cmap, norm=norm) 

ax.set_title('Rainfall (interpolated on the mask grid)',fontsize = 9.5)



#data with the target grid

ax = fig.add_subplot(spec[1,2])

fig.subplots_adjust(wspace=0, hspace=0)

label = 'Width: {}\nHeight: {}'.format(widths[2], heights[1])

ax.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')

plt.imshow(target_data[tar_param].values) 

ax.set_title('Land-sea mask (original grid)',fontsize = 9.5)
####Cell containing the modifiable fields######

date = '2016-02-01T00:00:00'    #study date 



obs_param = 't'      #observation parameter



#weather model parameters

model = 'arome' #weather model (arome or arpege)

MODEL = 'AROME' #weather model (AROME or ARPEGE)

level = '2m'      #vertical level (2m, 10m, P_sea_level or PRECIP)



grid_param = 't2m'   #AROME parameter

grid_time_step = 0  #index for the studied time step (index 0 corresponds to 00h00 in each weather model file, cf documentation)
study_date = pd.Timestamp(date)  #study date

fname = directory +zone+'_Ground_Stations/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations_'+str(year)+".csv"

df =pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)

d_sub = df[df['date'] == study_date]
display(d_sub.head())
directory_model = directory + zone + '_weather_models_2D_parameters_' + str(study_date.year) + str(study_date.month).zfill(2) + '/' + str(study_date.year) + str(study_date.month).zfill(2) + '/'

aro_fname = directory_model + f'{MODEL}/{level}/{model}_{level}_{zone}_{study_date.year}{str(study_date.month).zfill(2)}{str(study_date.day).zfill(2)}000000.nc'



aro = xr.open_dataset(aro_fname)

grid_lat = aro['latitude'].values

grid_lon = aro['longitude'].values

grid_val = aro[grid_param].values
def interpolate_grid_on_points(grid_lat,grid_lon,grid_val,data_obs):

    

    #initialisation

    latlon_grid = []

    latlon_obs = []

    val_grid = []

    

    #grid data preprocessing

    for i in range(0,grid_lat.shape[0]):        

        for j in range(0,grid_lon.shape[0]):

            #put coordinates (lat,lon) in list of tuples

            latlon_grid.append([round(grid_lat[i],3),round(grid_lon[j],3)])

            #put grid values into a list

            val_grid.append(grid_val[grid_time_step,i,j])

    grid_latlon = np.array(latlon_grid)

    grid_val2 = np.array(val_grid)



    #obs data preprocessing : put coordinates (lat,lon) in list of tuples

    for i in range(0,data_obs.shape[0]):

        latlon_obs.append([data_obs['lat'].values[i],data_obs['lon'].values[i]])

    latlon_obs = np.array(latlon_obs)

    

    #interpolation

    grid_val_on_points=griddata(grid_latlon ,grid_val2, latlon_obs,  method='linear')

    return latlon_obs,grid_val_on_points
latlon_obs,grid_val_on_points = interpolate_grid_on_points(grid_lat,grid_lon,grid_val,d_sub)
fig=plt.figure()

gs = gridspec.GridSpec(4, 4)



#Min and max boundaries about colorbar

vmin_obs = d_sub[obs_param].min()

vmax_obs = d_sub[obs_param].max()

vmin_model_ori= aro.isel(step=grid_time_step)[grid_param].values.min()

vmax_model_ori= aro.isel(step=grid_time_step)[grid_param].values.max()

vmin_model_inter=grid_val_on_points.min()

vmax_model_inter=grid_val_on_points.max()

vmin=np.min([vmin_obs,vmin_model_ori,vmin_model_inter])

vmax=np.max([vmax_obs,vmax_model_ori,vmax_model_inter])



#observation data

ax1 = plt.subplot(gs[:2, :2])

plt.tight_layout(pad=3.0)

im=ax1.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[obs_param], cmap='jet',vmin=vmin,vmax=vmax)

ax1.set_title('Observation data')



#weather model data (original grid)

ax2 = plt.subplot(gs[:2, 2:])

ax2.pcolor(grid_lon,grid_lat,aro.isel(step=grid_time_step)[grid_param].values,cmap="jet",vmin=vmin,vmax=vmax)

ax2.set_title('Weather model data (original grid)')



#weather model data (interpolated on observation points)

ax3 = plt.subplot(gs[2:4, 1:3])

ax3.scatter(latlon_obs[:,1], latlon_obs[:,0], c=grid_val_on_points, cmap='jet',vmin=vmin,vmax=vmax)

ax3.set_title('Weather model data (interpolated on observation points)')



fig.colorbar(im,ax=[ax1,ax2,ax3]).set_label('Temperature (in K)')

plt.show()
####Cell containing the modifiable fields######

###obs###

date = '2016-05-30T00:00:00'    #study date 

obs_param = 'hu'      #observation parameter

npz_param = 'rainfall'   #npz parameter



#rainfall##

year = 2016

month = 5

decade = 3

ind =  2592  #index of the chosen 5 min of the decade (index 2592 corresponds to 30/05 at 00h00 for the last decade, cf documentation)

nan_value = -1  #nan value for data (ex : rainfall here)



directory = '/kaggle/input/meteonet/'

rain_fname = directory + f'{zone}_rainfall_{str(year)}/{zone}_rainfall_{str(year)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall-{zone}-{str(year)}-{str(month).zfill(2)}/rainfall_{zone}_{str(year)}_{str(month).zfill(2)}.{str(part_month)}.npz'

rain_coords_fname = directory + 'Radar_coords/Radar_coords/'+ f'radar_coords_{zone}.npz'
study_date = pd.Timestamp(date)  #study date

fname = directory +zone+'_Ground_Stations/'+zone+'_Ground_Stations/'+zone+'_Ground_Stations_'+str(year)+".csv"

df =pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)

d_sub = df[df['date'] == study_date]
display(d_sub.head())
radar = np.load(rain_fname, allow_pickle=True)

data = radar['data'][ind,:,:]

coords = np.load(rain_coords_fname, allow_pickle=True)

#it is about coordinates of the center of pixels

lat = coords['lats']

lon = coords['lons']
def interpolate_radar_on_points(grid_lat,grid_lon,grid_val,data_obs):

    #grid data preprocessing

    latlon_grid = []

    latlon_obs = []

    val_grid = []

    for i in range(0,grid_lat.shape[0]):        

        for j in range(0,grid_lon.shape[1]):

            #put coordinates (lat,lon) in list of tuples

            latlon_grid.append([grid_lat[i,0],grid_lon[0,j]])

            #put grid values into a list

            val_grid.append(grid_val[i,j])

    grid_latlon = np.array(latlon_grid)

    grid_val2 = np.array(val_grid)

    #replace 'missing data' values by nan

    grid_val2 = grid_val2.astype(np.float64)

    grid_val2[grid_val2==-1]=np.nan



    #obs data preprocessing : put coordinates (lat,lon) in list of tuples

    for i in range(0,data_obs.shape[0]):

        latlon_obs.append([data_obs['lat'].values[i],data_obs['lon'].values[i]])

    latlon_obs = np.array(latlon_obs)



    #interpolation

    grid_val_on_points=griddata(grid_latlon ,grid_val2, latlon_obs,  method='linear')

    return latlon_obs,grid_val_on_points
latlon_obs,grid_val_on_points = interpolate_radar_on_points(lat,lon,data,d_sub)
fig=plt.figure()

gs = gridspec.GridSpec(4, 4)



data_with_nan = data.astype(np.float64).copy()

data_with_nan[data_with_nan==-1]=np.nan



#colorbar definition for rainfall

max_rr = max(np.nanmax(data),np.nanmax(d_sub[obs_param]),np.nanmax(grid_val_on_points))

if (max_rr > 65):

    borne_max = np.max(data_to_interpolate[ori_param].values)

else:

    borne_max = 65 + 10

cmap = colors.ListedColormap(['lavender','darkslateblue', 'mediumblue','dodgerblue', 'skyblue','olive','mediumseagreen'

                              ,'cyan','lime','yellow','khaki','burlywood','orange','brown','pink','red','plum'])

bounds = [0,1,4,6,8,10,15,20,25,30,35,40,45,50,55,60,65,borne_max]

norm = colors.BoundaryNorm(bounds, cmap.N)



#observation data

ax1 = plt.subplot(gs[:2, :2])

plt.tight_layout(pad=3.0)

im=ax1.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[obs_param], cmap = 'jet', s = 10)

ax1.set_title('Observation data : ' +obs_param + ' (%)')

plt.colorbar(im)



#rainfall data (original grid)

ax2 = plt.subplot(gs[:2, 2:])

ax2.pcolor(lon,lat,data_with_nan,cmap=cmap,norm=norm)

ax2.set_title('Radar : rainfall (original grid)')



#rainfall data (interpolated on observation points)

ax3 = plt.subplot(gs[2:4, 1:3])

im=ax3.scatter(latlon_obs[:,1], latlon_obs[:,0], c=grid_val_on_points, cmap=cmap,norm=norm, s = 10)

ax3.set_title('Radar : rainfall (interpolated on observation points)')



fig.colorbar(im,ax=[ax2,ax3]).set_label('Rainfall (in 1/100 mm) / NaN : missing values')

plt.show()