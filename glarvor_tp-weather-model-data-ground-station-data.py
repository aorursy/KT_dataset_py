import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
from datetime import timedelta  

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata


# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.
year = '2016'
fname = '/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+year+".csv"
df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
display(df.head())
display(df.tail())
date = '2016-01-01T06:00:00'
d_sub = df[df['date'] == date]

display(d_sub.head())
display(d_sub.tail())
param = 't'
plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')
plt.colorbar()
plt.title(date+' - param '+param)

plt.show()
# Coordinates of studied area boundaries (in °N and °E)
lllat = 46.25  #lower left latitude
urlat = 51.896  #upper right latitude
lllon = -5.842  #lower left longitude
urlon = 2  #upper right longitude
extent = [lllon, urlon, lllat, urlat]

fig = plt.figure(figsize=(9,5))

# Select projection
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot the data
plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')  # Plot
plt.colorbar()
plt.title(date+' - param '+param)


# Add coastlines and borders
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS.with_scale('50m'))

# Adjust the plot to the area we defined 
#/!\# this line causes a bug of the kaggle notebook and clears all the memory. That is why this line is commented and so
# the plot is not completely adjusted to the data
# Show only the area we defined
#ax.set_extent(extent)

plt.show()
def choose_parameters_and_display(date,param,df):
    #select the data corresponding to the selected date
    d_sub = df[df['date'] == date]
    
    # Coordinates of studied area boundaries (in °N and °E)
    lllat = 46.25  #lower left latitude
    urlat = 51.896  #upper right latitude
    lllon = -5.842  #lower left longitude
    urlon = 2  #upper right longitude
    extent = [lllon, urlon, lllat, urlat]

    fig = plt.figure(figsize=(9,5))

    # Select projection
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Plot the data
    plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')  # Plot
    plt.colorbar()
    plt.title(date+' - param '+param)

    # Add coastlines and borders
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    # Adjust the plot to the area we defined 
    #/!\# this line causes a bug of the kaggle notebook and clears all the memory. That is why this line is commented and so
    # the plot is not completely adjusted to the data
    # Show only the area we defined
    #ax.set_extent(extent)

    plt.show()
    return d_sub
date = '2016-12-31T12:24:00'
param = 'td'
data = choose_parameters_and_display(date,param,df)
#question 1

date = '2016-02-02T06:00:00'
station_id = 14066001
param = 't'
#%load /kaggle/usr/lib/tp_solutions_answer_obs_1/tp_solutions_answer_obs_1
#to execute the correction
#d_sub = obs_answer_1(year,date,station_id,param)
#question 2 

date = '2016-10-10T12:06:00'
station_id = 86137003
param = 'ff'
#to execute the correction
#d_sub = obs_answer_1(year,date,station_id,param)
#question 3 

date = '2016-12-25T15:12:00'
lat = 48.527
lon = 1.995
param = 'hu'
# %load /kaggle/usr/lib/tp_solutions_answer_obs_2/tp_solutions_answer_obs_2
#to execute the correction
#d_sub = obs_answer_2(year,date,station_id,param)
model = 'arome' #weather model (arome or arpege)
level = '2m'      #vertical level (2m, 10m, P_sea_level or PRECIP)
#date /!\ only available for February 2016!
date = dt.datetime(2016, 2, 14,0,0) #/!\ you can not modify the hour (always 00h) -> 1 possible run date only

directory = '/kaggle/input/meteonet/NW_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'
fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'
data = xr.open_dataset(fname)  
print(data)
param = 't2m'
data.isel(step=[0, 6, 12, 23])[param].plot(x='longitude',
                                           y='latitude',
                                           col='step',
                                           col_wrap=2)
coord = 'longitude'
print(data[coord])
data[coord].units
data[coord].values[0:10]
run_date = data['time']
#run_date.values     #get the values
print(run_date)
range_forecasts_dates = data['valid_time']
print(range_forecasts_dates)
#if you want information about vertical level
# if (level =='2m' or level == '10m'):
#     level_name = 'heightAboveGround'
# elif (level =='P_sea_level'):
#     level_name = 'meanSea'
# else:
#     level_name = 'surface'
# info_level = data[level_name]
# print(info_level)
d = data[param]     #param : parameter name defined at the beginning of the Notebook 
d_vals=d.values     #get the values
###examples to get the information from attributes
#d.units                      #unit
#d.long_name                      #long name
print(d)
print(d.dims)
print(d_vals.shape)
#question 1

run_date = '2016-02-10T00:00:00'
param = 't2m'  #cf the cell above to know the parameter names in the observation file 
model = 'arome' #weather model (arome or arpege)
lat = 51.696
lon = 0.008
step = 3
# %load /kaggle/usr/lib/tp_solutions_answer_mod/tp_solutions_answer_mod
#to execute the correction
#result1 = open_and_select(run_date,param,model,lat,lon,step)
#question 2

run_date = '2016-02-01T00:00:00'
param = 'ws'  #cf the cell above to know the parameter names in the observation file 
model = 'arpege' #weather model (arome or arpege)
lat = 48.896
lon = 0.558
step = 6

#to execute the correction
#result2 = open_and_select(run_date,param,model,lat,lon,step)
#question 3 

run_date = '2016-02-20T00:00:00'
param = 'tp'  #cf the cell above to know the parameter names in the observation file 
model = 'arpege' #weather model (arome or arpege)
lat = 47.496
lon = 1.858
step = 19

#to execute the correction
#result3 = open_and_select(run_date,param,model,lat,lon,step)
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
plt.title(model +" model - "+str(d['valid_time'].values[step])+" - " +"NW zone")
plt.show()
def choose_parameters_and_display(model,run_date,step,param):
    
    #open the corresponding file according to the chosen parameter    
    if param == 't2m' or param == 'd2m' or param == 'r':
        level = '2m'
    elif param == 'ws' or param =='p3031' or param == 'u10' or param == 'v10':
        level = '10m'
    elif param == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'
    
    directory = '/kaggle/input/meteonet/NW_weather_models_2D_parameters_' + str(run_date.year) + str(run_date.month).zfill(2) + '/' + str(run_date.year) + str(run_date.month).zfill(2) + '/'
    fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{run_date.year}{str(run_date.month).zfill(2)}{str(run_date.day).zfill(2)}000000.nc'
    sub_data = xr.open_dataset(fname)      
    
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
    img = ax.imshow(sub_data[param].values[step,:,:], interpolation='none', origin='upper', extent=extent)
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))


    plt.colorbar(img, orientation= 'horizontal').set_label(sub_data[param].long_name+ ' (in '+sub_data[param].units+ ')')
    plt.title(model +" model - "+str(sub_data['valid_time'].values[step])+" - " +"NW zone")
    plt.show()
    
    return sub_data
model = 'arpege' #weather model (arome or arpege)
#run date /!\ only available for February 2016!
run_date = dt.datetime(2016, 2,10,0,0) #/!\ you can not modify the hour (always 00h) -> 1 possible run date only
param = 'ws'    #parameter name in the file (cf correspondences in the cell below)
step = 3   #index of chosen time step (from 0 to 24 and about precipitation, from 0 to 23)

sub_data = choose_parameters_and_display(model,run_date,step,param)
#date
date = '2016-02-13' 

#observation
param_obs = 'dd'  #cf the cell below to know the parameter names in the observation file 

#model
model = 'arpege' #weather model (arome or arpege)
MODEL = 'ARPEGE' #weather model (AROME or ARPEGE)
param_mod = 'p3031'   #cf correspondences in the cell below

#algorithm about interpolation
algo = 'linear' #or 'nearest' for nearest neighbors
#first part : open and filtering on the day
year = date[0:4]
fname = '/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+year+".csv"

def open_and_date_filtering(year,fname,date):    
    #open the observation data
    #df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True) #execution time ~1 min 

    #filtering on the date 
    study_date = pd.Timestamp(date)  #study date
    d_sub = df[(df['date'] >= study_date) & (df['date'] <= study_date + timedelta(days=1))]
    d_sub = d_sub.set_index('date')
    display(d_sub.head())
    display(d_sub.tail())
    return(d_sub)

d_sub = open_and_date_filtering(year,fname,date)
#second part : choose the station_id, get the lat/lon of the station and resample the data to the hourly step!
station_id = 86137003

def station_lat_lon_resample(station_id,d_sub):
    #filtering on the station_id
    d_sub = d_sub[d_sub['number_sta'] == station_id]

    #get the lat/lon values of the station 
    lat_sta = d_sub['lat'][0]
    lon_sta = d_sub['lon'][0]

    #resample the 6 min data to get hourly data (by using the mean on one hour)
    d_sub = d_sub[param_obs].resample('H').mean()
    print('station_id:',station_id)
    print('lat/lon:',lat_sta,'/',lon_sta)
    print('weather parameter',param_obs)
    display(d_sub)
    return(d_sub,station_id,lat_sta,lon_sta)

d_sub, station_id, lat_sta, lon_sta = station_lat_lon_resample(station_id,d_sub)
def open_get_values(param_mod,date):
    #open the corresponding file according to the chosen parameter
    if param_mod == 't2m' or param_mod == 'd2m' or param_mod == 'r':
        level = '2m'
    elif param_mod == 'ws' or param_mod =='p3031' or param_mod == 'u10' or param_mod == 'v10':
        level = '10m'
    elif param_mod == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'

    year = date[0:4]
    month = date[5:7]
    day = date[8:10]

    directory = '/kaggle/input/meteonet/NW_weather_models_2D_parameters_' + year + month + '/' + year + month + '/'
    fname = directory + f'{MODEL}/{level}/{model}_{level}_NW_{year}{month}{day}000000.nc'
    dm = xr.open_dataset(fname)
    print('dataset overview:',dm)

    #get the forecast values and the lat/lon of the grid data
    grid_values = dm[param_mod].values
    grid_lat = dm['latitude'].values
    grid_lon = dm['longitude'].values
    print('shape of the forecast values array:',grid_values.shape)
    print('ten first latitudes:',grid_lat[0:10])
    print('ten first longitudes',grid_lon[0:10])
    return(grid_values,grid_lat,grid_lon)
grid_values,grid_lat,grid_lon = open_get_values(param_mod,date)
from scipy.interpolate import griddata

def interpolation(grid_values,grid_lat,grid_lon,lat_sta,lon_sta):
    #initialization
    model_values = []
    grid_on_points = np.empty(grid_values.shape[0], dtype = object) 
    
    #loop per time step 
    for step in range(0,grid_values.shape[0]):
        latlon_grid = []
        val_grid = []
        latlon_obs = []

        #grid data preprocessing
        for i in range(0,grid_lat.shape[0]):        
            for j in range(0,grid_lon.shape[0]):
                #put coordinates (lat,lon) in list of tuples
                latlon_grid.append([grid_lat[i],grid_lon[j]])
                #put grid values into a list
                val_grid.append(grid_values[step,i,j])

        grid_latlon = np.array(latlon_grid)
        grid_val2 = np.array(val_grid)

        #ground station position (lat/lon) preprocessing
        latlon_obs.append([lat_sta,lon_sta])
        latlon_obs = np.array(latlon_obs)

        #compute the interpolation
        grid_on_points[step] = griddata(grid_latlon ,grid_val2, latlon_obs,  method=algo)[0]
        print('step ',step, ' OK!')
    return(grid_on_points)
grid_on_points = interpolation(grid_values,grid_lat,grid_lon,lat_sta,lon_sta)
obs = d_sub
def preproc_output(obs,grid_on_points,param_mod):
    mod = pd.Series(grid_on_points,index=obs.index)
    print('interpolated forecasted data, param ',param_mod)
    return (mod)
mod = preproc_output(obs,grid_on_points,param_mod)
display(mod)
def plots(obs,mod,MODEL,param_obs,lat_sta,lon_sta):
    plt.plot(obs, label ='Observation')
    plt.plot(mod, label = MODEL +' forecast')
    plt.title('Parameter '+param_obs+' / lat='+str(lat_sta)+' and lon='+str(lon_sta))
    plt.xlabel('Time')
    plt.ylabel(param_obs)
    plt.legend()
plots(obs,mod,MODEL,param_obs,lat_sta,lon_sta)
#observation
date_obs = '2016-02-10T10:00:00' 
param_obs = 'ff'

#model
model = 'arpege' #weather model (arome or arpege)
MODEL = 'ARPEGE' #weather model (AROME or ARPEGE)
date_mod = dt.datetime(2016, 2,10,10,0) # Day example 
param_mod = 'ws'

#algorithm about interpolation
algo = 'linear' #or 'nearest' for nearest neighbors
fname = '/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+date_obs[0:4]+".csv"
#df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
study_date = pd.Timestamp(date_obs)  #study date
d_sub = df[df['date'] == study_date]
print('observation data',d_sub)
directory = '/kaggle/input/meteonet/NW_weather_models_2D_parameters_' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/'

if param_mod == 't2m' or param_mod == 'd2m' or param_mod == 'r':
    level = '2m'
elif param_mod == 'ws' or param_mod =='p3031' or param_mod == 'u10' or param_mod == 'v10':
    level = '10m'
elif param_mod == 'msl':
    level = 'P_sea_level'
else:
    level = 'PRECIP'

fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{date_mod.year}{str(date_mod.month).zfill(2)}{str(date_mod.day).zfill(2)}000000.nc'

mod = xr.open_dataset(fname)
grid_lat = mod['latitude'].values
grid_lon = mod['longitude'].values
grid_val = mod[param_mod].values[date_mod.hour,:,:]
print('latitudes on the model grid:',grid_lat)
print('longitudes on the model grid:',grid_lon)
print('forecast values:',grid_val)
def interpolate_grid_on_points(grid_lat,grid_lon,grid_val,data_obs,algo):
    
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
            val_grid.append(grid_val[i,j])
    grid_latlon = np.array(latlon_grid)
    grid_val2 = np.array(val_grid)

    #obs data preprocessing : put coordinates (lat,lon) in list of tuples
    for i in range(0,data_obs.shape[0]):
        latlon_obs.append([data_obs['lat'].values[i],data_obs['lon'].values[i]])
    latlon_obs = np.array(latlon_obs)
    
    #interpolation
    grid_val_on_points=griddata(grid_latlon ,grid_val2, latlon_obs,  method=algo)
    return latlon_obs,grid_val_on_points
latlon_obs,grid_val_on_points = interpolate_grid_on_points(grid_lat,grid_lon,grid_val,d_sub,algo)
print('10 first lat/lon couple per station:',latlon_obs[0:10,:])
print('associated forecast values interpolated on ground station points:',grid_val_on_points[0:10])
#Plot the different data
fig=plt.figure()
gs = gridspec.GridSpec(4, 4)

#Min and max boundaries about colorbar
vmin_obs = d_sub[param_obs].min()
vmax_obs = d_sub[param_obs].max()
vmin_model_ori= grid_val.min()
vmax_model_ori= grid_val.max()
vmin_model_inter=grid_val_on_points.min()
vmax_model_inter=grid_val_on_points.max()
vmin=np.min([vmin_obs,vmin_model_ori,vmin_model_inter])
vmax=np.max([vmax_obs,vmax_model_ori,vmax_model_inter])

#observation data
ax1 = plt.subplot(gs[:2, :2])
plt.tight_layout(pad=3.0)
im=ax1.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param_obs], cmap='jet',vmin=vmin,vmax=vmax)
ax1.set_title('Observation data')

#weather model data (original grid)
ax2 = plt.subplot(gs[:2, 2:])
ax2.pcolor(grid_lon,grid_lat,grid_val,cmap="jet",vmin=vmin,vmax=vmax)
ax2.set_title('Weather model data (original grid)')

#weather model data (interpolated on observation points)
ax3 = plt.subplot(gs[2:4, 1:3])
im3=ax3.scatter(latlon_obs[:,1], latlon_obs[:,0], c=grid_val_on_points, cmap='jet',vmin=vmin,vmax=vmax)
ax3.set_title('Weather model data (interpolated on observation points)')

fig.colorbar(im,ax=[ax2,ax3]).set_label(mod[param_mod].long_name+ ' (in '+mod[param_mod].units+ ')')
plt.show()
def choose_and_display(model,date_mod,date_obs,param_obs,param_mod,algo,d_sub):
    
    #get the observation values 
    study_date = pd.Timestamp(date_obs)  #study date
    d_sub = df[df['date'] == study_date]
    
    #get the model data 
    directory = '/kaggle/input/meteonet/NW_weather_models_2D_parameters_' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/'

    if param_mod == 't2m' or param_mod == 'd2m' or param_mod == 'r':
        level = '2m'
    elif param_mod == 'ws' or param_mod =='p3031' or param_mod == 'u10' or param_mod == 'v10':
        level = '10m'
    elif param_mod == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'

    fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{date_mod.year}{str(date_mod.month).zfill(2)}{str(date_mod.day).zfill(2)}000000.nc'

    mod = xr.open_dataset(fname)
    grid_lat = mod['latitude'].values
    grid_lon = mod['longitude'].values
    grid_val = mod[param_mod].values[date_mod.hour,:,:]
    
    #perform the interpolation
    latlon_obs,grid_val_on_points = interpolate_grid_on_points(grid_lat,grid_lon,grid_val,d_sub,algo)
    
    #Plot the different data
    fig=plt.figure()
    gs = gridspec.GridSpec(4, 4)

    #Min and max boundaries about colorbar
    vmin_obs = d_sub[param_obs].min()
    vmax_obs = d_sub[param_obs].max()
    vmin_model_ori= grid_val.min()
    vmax_model_ori= grid_val.max()
    vmin_model_inter=grid_val_on_points.min()
    vmax_model_inter=grid_val_on_points.max()
    vmin=np.min([vmin_obs,vmin_model_ori,vmin_model_inter])
    vmax=np.max([vmax_obs,vmax_model_ori,vmax_model_inter])

    #observation data
    ax1 = plt.subplot(gs[:2, :2])
    plt.tight_layout(pad=3.0)
    im=ax1.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param_obs], cmap='jet',vmin=vmin,vmax=vmax)
    ax1.set_title('Observation data')

    #weather model data (original grid)
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.pcolor(grid_lon,grid_lat,grid_val,cmap="jet",vmin=vmin,vmax=vmax)
    ax2.set_title('Weather model data (original grid)')

    #weather model data (interpolated on observation points)
    ax3 = plt.subplot(gs[2:4, 1:3])
    im3=ax3.scatter(latlon_obs[:,1], latlon_obs[:,0], c=grid_val_on_points, cmap='jet',vmin=vmin,vmax=vmax)
    ax3.set_title('Weather model data (interpolated on observation points)')

    fig.colorbar(im,ax=[ax2,ax3]).set_label(mod[param_mod].long_name+ ' (in '+mod[param_mod].units+ ')')
    plt.show()
    
    return d_sub, mod[param_mod][date_mod.hour,:,:], latlon_obs,grid_val_on_points
#observation
date_obs = '2016-02-10T10:00:00' 
param_obs = 'hu'

#model
model = 'arome' #weather model (arome or arpege)
date_mod = dt.datetime(2016, 2,10,10,0) # Day example 
param_mod = 'r'

#algorithm about interpolation
algo = 'nearest' #'linear' or 'nearest' for nearest neighbors
obs_output, mod_output, latlon_obs,grid_val_on_points =  choose_and_display(model,date_mod,date_obs,param_obs,param_mod,algo,d_sub)