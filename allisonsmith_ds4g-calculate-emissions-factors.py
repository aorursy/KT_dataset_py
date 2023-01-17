##Shell Script

##requires gdal, nco, cdo to run 

##change directory

# cd ../data/starter_pack/s5p_no2/



# filenamelist=`ls tif/*.tif`

# for tif_file in $filenamelist

# do



#     echo $tif_file

#     #tif_file="tif/s5p_no2_20180701T161259_20180707T175356.tif"



#     #set names

#     filename=${tif_file%.tif}

#     nc_file=nc/${filename#tif/}.nc



#     #convert tif to netCDF

#     gdal_translate -of NETCDF -co "FORMAT=NC4" ${tif_file} convert.nc



#     #rename th variables in the netcdf files

#     ncrename -v Band1,NO2_column_number_density convert.nc

#     ncrename -v Band2,tropospheric_NO2_column_number_density convert.nc

#     ncrename -v Band3,stratospheric_NO2_column_number_density convert.nc

#     ncrename -v Band4,NO2_slant_column_number_density convert.nc

#     ncrename -v Band5,tropopause_pressure convert.nc

#     ncrename -v Band6,absorbing_aerosol_index convert.nc

#     ncrename -v Band7,cloud_fraction convert.nc

#     ncrename -v Band8,sensor_altitude convert.nc

#     ncrename -v Band9,sensor_azimuth_angle convert.nc

#     ncrename -v Band10,sensor_zenith_angle convert.nc

#     ncrename -v Band11,solar_azimuth_angle convert.nc

#     ncrename -v Band12,solar_zenith_angle convert.nc



#     #extract time from the file name and add as a dimension to the netCDF file

#     timestrT="$(cut -d'_' -f3 <<<${filename})"

#     timestr="${timestrT//T}"

#     unix="$(date -j -u -f "%Y%m%d%H%M%S" "${timestr}" +"%s")"

#     # printf "/***.nco ncap2 script***/

#     # defdim(\"time\",1);

#     # time[time]="$unix";

#     # time@long_name=\"Time\";

#     # time@units=\"seconds since 1970-01-01 00:00:00\";

#     # time@standard_name=\"time\";

#     # time@axis=\"T\";

#     # time@coordinate_defines=\"point\";

#     # time@calendar=\"standard\";

#     # /***********/" > time.nco



#     ncap2 -Oh -s 'defdim("time",1);time[time]=1561830483;time@long_name="Time";time@units="seconds since 1970-01-01 00:00:00";time@standard_name="time";time@axis="T";time@coordinate_defines="point";time@calendar="standard";' convert.nc convert1.nc

#     date_string="$(date -u -r ${unix} +'%Y-%m-%d,%H:%M:%S')"

#     cdo settaxis,"${date_string}" convert1.nc convert2.nc



#     #save final version of netcdf file 

#     mv convert2.nc ${nc_file}



#     #clean up files

#     rm convert.nc

#     rm convert1.nc

#     #rm time.nco 



# done



# cdo mergetime nc/*.nc no2_1year.nc
import numpy as np

import pandas as pd

import xarray as xr

#import xesmf as xe

import json

import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import seaborn as sns

import json

import datetime as dt

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#Reusable function for doing the regridding:

# def make_regridder(ds, ds_base, variable, algorithm='bilinear'):   

#     if 'latitude' in ds[variable].dims:

#        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'}).set_coords(['lon', 'lat'])

#     ds_regrid = xr.Dataset({'lat': (['lat'], np.arange(np.floor(ds_base['lat'].min().values*10)/10, np.ceil(ds_base['lat'].max().values*10)/10, 0.01)),

#                      'lon': (['lon'], np.arange(np.floor(ds_base['lon'].min().values*10)/10, np.ceil(ds_base['lon'].max().values*10)/10, 0.01)),

#                     }

#                    )



#     regridder = xe.Regridder(ds, ds_regrid, algorithm)

#     regridder.clean_weight_file()

#     return regridder
#Base grid is based on the s5p_no2 data, i.e. all other data sets will be regrid to the no2 grid

# ds_s5p = xr.open_dataset('../data/starter_pack/s5p_no2/no2_1year.nc')

# ds_no2_clouds = ds_s5p[['NO2_column_number_density', 'cloud_fraction']]

# no2_regridder = make_regridder(ds_no2_clouds, ds_no2_clouds, 'NO2_column_number_density')

# ds_base_regrid = no2_regridder(ds_no2_clouds)

# ds_base_regrid = ds_base_regrid.where(ds_base_regrid['NO2_column_number_density']!=0.)
#Plot the NO_2 data for a single time point

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base_regrid.NO2_column_number_density.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree());

# ax.coastlines()

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
#Download Super High Resolution GHRSST SST file (0.01 degree grid)

#https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplG1SST.nc?SST[(2017-09-13T00:00:00Z):1:(2017-09-13T00:00:00Z)][(17.005):1:(19.005)][(-69.995):1:(-64.005)],mask[(2017-09-13T00:00:00Z):1:(2017-09-13T00:00:00Z)][(17.005):1:(19.005)][(-69.995):1:(-64.005)],analysis_error[(2017-09-13T00:00:00Z):1:(2017-09-13T00:00:00Z)][(17.005):1:(19.005)][(-69.995):1:(-64.005)]
#Read in SST data, the file is in the Kaggle folder input/ds4g-emissions-nc

# ds_sea = xr.open_dataset('../input/ds4g-emissions-nc/jplG1SST_e435_8209_9395.nc')
#Plot SST to view the grid

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_sea.SST.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree());

# ax.coastlines()

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
#Regrid SST data to the same grid as the NO2 data

# sea_regridder = make_regridder(ds_sea, ds_base_regrid, 'SST')

# ds_sea_regrid = sea_regridder(ds_sea)

# ds_sea_regrid = ds_sea_regrid.where(ds_sea_regrid['SST']!=0.)
#Plot the regridded data

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_sea_regrid.SST.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree());

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
#Create a land mask and add it to the base data set

# land_ones = ds_sea_regrid.SST.isel(time=0).fillna(1)

# land_mask = land_ones.where(land_ones ==1.)

# land_mask = land_mask.where(land_mask.lat<18.5)

# land_mask = land_mask.drop('time')

# ds_base_regrid.coords['land_mask'] = land_mask
#Plot NO2 for just the land

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base_regrid['NO2_column_number_density'].isel(time=0).where(ds_base_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_base = ds_base_regrid.resample(time='1D').mean()
#Note the changes to the timestamp when the data are plotted

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base['NO2_column_number_density'].isel(time=0).where(ds_base.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_gldas = xr.open_dataset('../data/starter_pack/gldas/gldas_1year.nc')

# ds_gldas = ds_gldas.drop('crs')

# gldas_regridder = make_regridder(ds_gldas, ds_base_regrid, 'Tair_f_inst',  'nearest_s2d')

# ds_gldas_regrid = gldas_regridder(ds_gldas)

# ds_gldas_regrid = ds_gldas_regrid.where(ds_gldas_regrid['Tair_f_inst']!=0.)

# ds_gldas_regrid.coords['land_mask'] = land_mask
# ds_gldas_regrid_fill = ds_gldas_regrid.ffill(dim='lat')

# ds_gldas_regrid_fill = ds_gldas_regrid_fill.bfill(dim='lat')

# ds_gldas_regrid_fill = ds_gldas_regrid_fill.ffill(dim='lon')

# ds_gldas_regrid_fill = ds_gldas_regrid_fill.bfill(dim='lon')
# #The gldas data without the coastal pixels filled

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_gldas_regrid['Tair_f_inst'].isel(time=0).where(land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# #The gldas data with the coastal pixels filled

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_gldas_regrid_fill['Tair_f_inst'].isel(time=0).where(land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_gldas_daily_mean = ds_gldas_regrid_fill.resample(time='1D').mean()

# ds_gldas_daily_max = ds_gldas_regrid_fill.resample(time='1D').max()

# ds_gldas_daily_min = ds_gldas_regrid_fill.resample(time='1D').min()
# ds_base['gldas_wind_mean'] = ds_gldas_daily_mean['Wind_f_inst']

# ds_base['gldas_airT_mean'] = ds_gldas_daily_mean['Tair_f_inst']

# ds_base['gldas_airT_max'] = ds_gldas_daily_max['Tair_f_inst']

# ds_base['gldas_airT_min'] = ds_gldas_daily_min['Tair_f_inst']

# ds_base['gldas_lwdown_mean'] = ds_gldas_daily_mean['LWdown_f_tavg']

# ds_base['gldas_pres_mean'] = ds_gldas_daily_mean['Psurf_f_inst']

# ds_base['gldas_humidity_mean'] = ds_gldas_daily_mean['Qair_f_inst']

# ds_base['gldas_heatflux_mean'] = ds_gldas_daily_mean['Qg_tavg']

# ds_base['gldas_rain_max'] = ds_gldas_daily_max['Rainf_f_tavg']

# ds_base['gldas_SWdown_max'] = ds_gldas_daily_max['SWdown_f_tavg']
# ds_gfs = xr.open_dataset('../data/starter_pack/gfs/gfs_1year.nc')

# ds_gfs = ds_gfs.drop('crs')

# gfs_regridder = make_regridder(ds_gfs, ds_base_regrid, 'temperature_2m_above_ground')

# ds_gfs_regrid = gfs_regridder(ds_gfs)

# ds_gfs_regrid = ds_gfs_regrid.where(ds_gfs_regrid['temperature_2m_above_ground']!=0.)

# ds_gfs_regrid.coords['land_mask'] = land_mask

# ds_gfs_regrid['wind_speed'] = np.sqrt(np.square(ds_gfs_regrid.u_component_of_wind_10m_above_ground) + np.square(ds_gfs_regrid.v_component_of_wind_10m_above_ground))
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_gfs_regrid['wind_speed'].isel(time=6).where(ds_gfs_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_gfs_daily_mean = ds_gfs_regrid.resample(time='1D').mean()

# ds_base['gfs_wind_mean'] = ds_gfs_daily_mean['wind_speed']
##Data were downloaded using the Google Earth Engine Interface.  

# // Create a geometry representing an export region.

# var geometry = ee.Geometry.Rectangle([-67.4, 17.9, -65.1, 18.6])



# // Load an image

# var getimagecollection = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")

#       .filterDate('2018-07-01', '2019-07-01')

#       .filterBounds(geometry);



# var imageselect = getimagecollection.select('avg_rad').first();



# // var getimage = ee.Image(imageselect).first();





# // Export the image, specifying scale and region.

# Export.image.toDrive({

# image: imageselect,

# description: 'ds4g_nighttime_lights2',

# scale: 1000,

# region: geometry,

# });
# ds_nightlights = xr.open_dataset('../input/ds4g-emissions-nc/VIIRS_nighttime_lights.nc')

# ds_nightlights2 = ds_nightlights.drop('crs')
# nl_regridder = make_regridder(ds_nightlights2, ds_base_regrid, 'avg_rad')



# ds_nl_regrid = nl_regridder(ds_nightlights2)

# ds_nl_regrid = ds_nl_regrid.where(ds_nl_regrid['avg_rad']!=0.)

# ds_nl_regrid.coords['land_mask'] = land_mask
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_nl_regrid['avg_rad'].where(ds_nl_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_base['night_avg_rad'] = ds_nl_regrid['avg_rad']
##Data were downloaded using the Google Earth Engine Interface.  

# // Create a geometry representing an export region.

# var geometry = ee.Geometry.Rectangle([-67.4, 17.9, -65.1, 18.6])



# // Load an image

# var getimage = ee.Image("USGS/GFSAD1000_V1")



# var imageselect = getimage.select('landcover');



# // var getimage = ee.Image(imageselect).first();





# // Export the image, specifying scale and region.

# Export.image.toDrive({

# image: imageselect,

# description: 'ds4g_landcover',

# scale: 1000,

# region: geometry,

# });
# ds_landcover = xr.open_dataset('../input/ds4g-emissions-nc/GFSAD1000_landcover.nc')

# ds_landcover = ds_landcover.drop('crs')

# land_regridder = make_regridder(ds_landcover, ds_base_regrid, 'landcover_category',  'nearest_s2d')

# ds_land_regrid = land_regridder(ds_landcover)

# ds_land_regrid.coords['land_mask'] = land_mask
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_land_regrid['landcover_category'].where(ds_land_regrid.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_base['landcover_category'] = ds_land_regrid['landcover_category']
#Read in the data 

# plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

# plants = plants[['capacity_mw', 'estimated_generation_gwh', 'primary_fuel', '.geo']]



#Extract the latitude and longitude coordinates from a json string into separate columns

# coordinates = pd.json_normalize(plants['.geo'].apply(json.loads))['coordinates']

# plants[['longitude', 'latitude']] = pd.DataFrame(coordinates.values.tolist(), index= coordinates.index)

# plants.drop('.geo', axis=1, inplace=True)





# pd.json_normalize was missing from Kaggle, I uploaded the data with latitude and longitude as separate columns

plants = pd.read_csv('../input/ds4g-emissions-nc/gppd_120_pr_lat_lon.csv')

#Filter for plants that burn fossil fuels and generate NO2

plants_fossil = plants[plants['primary_fuel'].isin(['Oil', 'Gas', 'Coal'])].copy()

plants_fossil.reset_index(drop=True, inplace=True)

# plants_fossil['grid_lon'] = np.nan

# plants_fossil['position_lon'] = np.ones

# plants_fossil['grid_lat'] = np.nan

# plants_fossil['position_lat'] = np.ones



#Map the fossil fuel plants to the nearest grid cells in the base grid 

# lons = ds_base.lon.values

# a=0 

# for lon in plants_fossil.longitude:

#     lon_diff = abs(lon-lons) 

#     plants_fossil.at[a,'grid_lon'] = lons[np.argmin(lon_diff)]

#     plants_fossil.at[a,'position_lon'] = np.argmin(lon_diff)

#     a=a+1



# lats = ds_base.lat.values

# a=0 

# for lat in plants_fossil.latitude:

#     lat_diff = abs(lat-lats) 

#     plants_fossil.at[a,'grid_lat'] = lats[np.argmin(lat_diff)]

#     plants_fossil.at[a,'position_lat'] = np.argmin(lat_diff)

#     a=a+1
#Calculate the number of plants in each grid cell

plants_fossil['num_plants'] = 1

#plants_fossil_grid = plants_fossil[['grid_lon', 'grid_lat', 'position_lat', 'position_lon', 'num_plants']].groupby(['grid_lon', 'grid_lat', 'position_lat', 'position_lon'], as_index=False).sum()

plants_fossil_grid = pd.read_csv('../input/ds4g-emissions-nc/plants_fossil_grid.csv')
#Save data for future use

#plants_fossil.to_csv('plants_fossil.csv', index=False)
# Convert data frame into a grid that can be added as a mask in the base grid

# Also creating a mask with the positions of the latitude and longitudes in the grid to use for the marginal emissions analysis

# plants_mask = 0 * np.ones((ds_base.dims['lat'], ds_base.dims['lon'])) * np.isnan(ds_base.NO2_column_number_density.isel(time=0)) 

# position_lat_id = 0 * np.ones((ds_base.dims['lat'], ds_base.dims['lon'])) * np.isnan(ds_base.NO2_column_number_density.isel(time=0))

# position_lon_id = 0 * np.ones((ds_base.dims['lat'], ds_base.dims['lon'])) * np.isnan(ds_base.NO2_column_number_density.isel(time=0))

# plants_mask = plants_mask.drop('time')



# # Create masks for the fossil fuel power plants, 

# # The mask also includes the grid cells immediately surround the grid cell with the power plant

# for x in plants_fossil_grid.index:

#     plants_mask[(plants_fossil_grid.at[x,'position_lat']-2):(plants_fossil_grid.at[x,'position_lat']+2),(plants_fossil_grid.at[x,'position_lon']-2):(plants_fossil_grid.at[x,'position_lon']+2)]=1

#     position_lat_id[(plants_fossil_grid.at[x,'position_lat']-2):(plants_fossil_grid.at[x,'position_lat']+2),(plants_fossil_grid.at[x,'position_lon']-2):(plants_fossil_grid.at[x,'position_lon']+2)]=plants_fossil_grid.at[x,'position_lat']

#     position_lon_id[(plants_fossil_grid.at[x,'position_lat']-2):(plants_fossil_grid.at[x,'position_lat']+2),(plants_fossil_grid.at[x,'position_lon']-2):(plants_fossil_grid.at[x,'position_lon']+2)]=plants_fossil_grid.at[x,'position_lon']



# # Add the masks to the base grid array    

# plants_mask = plants_mask.where(plants_mask == 1.)

# position_lat_id = position_lat_id.where(position_lat_id >= 1.)

# position_lon_id = position_lon_id.where(position_lon_id >= 1.)

# ds_base.coords['plants_mask'] = (('lat', 'lon'), plants_mask)

# ds_base.coords['no_plants_mask'] = ds_base.plants_mask.fillna(0).where((ds_base.plants_mask != 1) & (ds_base.land_mask == 1))

# ds_base.coords['no_plants_mask']  = ds_base.no_plants_mask + 1

# ds_base.coords['position_lat_id'] = (('lat', 'lon'), position_lat_id)

# ds_base.coords['position_lat_id'] = ds_base.position_lat_id.where(ds_base.position_lat_id >= 1)

# ds_base.coords['position_lon_id'] = (('lat', 'lon'), position_lon_id)

# ds_base.coords['position_lon_id'] = ds_base.position_lon_id.where(ds_base.position_lon_id >= 1)

# Plot the grid cell areas with power plants 

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base['NO2_column_number_density'].isel(time=0).where((land_mask==1) & (plants_mask==1)).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# Plot the grid cell ares without power plants

# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base['NO2_column_number_density'].isel(time=0).where(ds_base.no_plants_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
# ds_base_annual = ds_base.where((ds_base.gfs_wind_mean <= 2)).mean(dim=['time'])
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base_annual['NO2_column_number_density'].where((ds_base_annual.land_mask == 1) & (ds_base_annual.no_plants_mask ==1)).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
#Base grid file is finished. 

#Save a copy to a NetCDF file to be used to calculate the emissions factor.

#ds_base_annual.to_netcdf('annual_ds4g_emissions.nc')
# ds_base_monthly = ds_base.where((ds_base.gfs_wind_mean <= 5)).resample(time='1M').mean()
# ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

# ds_base_monthly['NO2_column_number_density'].isel(time=0).where(ds_base_monthly.land_mask == 1).plot(ax=ax, transform=ccrs.PlateCarree())

# ax.set_extent([-67.5, -65, 17.5, 19])

# ax.set_aspect("equal")
#Save a copy to a NetCDF file to be used to calculate the emissions factor for each month.

#ds_base_monthly.to_netcdf('monthly_ds4g_emissions.nc')
#Read in the data set from the saved file created using the code in the regridding section above.

ds = xr.open_dataset('../input/ds4g-emissions-nc/annual_ds4g_emissions.nc')



#Use data set created above.

#ds = ds_base_annual
print(ds)
#Create a plot of the NO2 emissions from the data set

ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

ds.NO2_column_number_density.where(ds.land_mask==1).plot(ax=ax, transform=ccrs.PlateCarree());
#Create a plot of the gfs wind speed from the data set

ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

ds.wind_speed_mean.where(ds.land_mask==1).plot(ax=ax, transform=ccrs.PlateCarree());
#Create a plot of the NO2 emissions for just the power plant grid cells

ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

ds.NO2_column_number_density.where(ds.plants_mask==1).plot(ax=ax, transform=ccrs.PlateCarree());
ds['landcover_category'] = ds['landcover_category'].fillna(0)

ds_land = ds.where((ds.land_mask == 1) & 

                   (ds.no_plants_mask == 1) & 

                   (ds.landcover_category == 0))
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-65, central_latitude=18))

ds_land.NO2_column_number_density.plot(ax=ax, transform=ccrs.PlateCarree());
ds_vehicles = ds_land

ds_vehicles = ds_vehicles.drop(['no_plants_mask','plants_mask','land_mask', 'position_lat_id', 'position_lon_id'])

df_vehicles = ds_vehicles.to_dataframe().dropna()
# Explore features

# df_vehicles.describe()

# sns.pairplot(df_vehicles)
df_vehicles = df_vehicles[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean']].copy()

df_vehicles['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_vehicles['cloud_fraction']))

df_vehicles['log_night_avg_rad'] = np.log(df_vehicles['night_avg_rad'])

df_vehicles.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)
# Generate a plot to make sure that none of the variables are highly correlated. 



df_vehicles2 = df_vehicles.copy()

df_vehicles2['NO2_column_number_density'] = df_vehicles['NO2_column_number_density']*(10**5) #Kaggle does not always plot very small numbers so rescaling.

sns.pairplot(df_vehicles2)
X = np.array(df_vehicles.drop(['NO2_column_number_density'], axis=1))

y = np.array(df_vehicles['NO2_column_number_density'])

y = y*(10**5) #The numbers are very small so rescaling for model training and prediction

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=122)

model = XGBRegressor(learning_rate = 0.01, objective='reg:squarederror', n_estimators = 500)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)

y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred) #Kaggle has limits on small scales, so these should be 10**-5
print(r2_score(y_train, y_train_pred))

print(r2_score(y_test, y_pred))

print(mean_squared_error(y_test*10**-5, y_pred*10**-5))

print(mean_absolute_error(y_test*10**-5, y_pred*10**-5))
#Create a data frame with only the power plant grid cells

ds_powerplants = ds.where(ds.plants_mask == 1)

ds_powerplants = ds_powerplants.drop(['no_plants_mask','plants_mask','land_mask'])

df_powerplants = ds_powerplants.to_dataframe().dropna()

df_powerplants = df_powerplants[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean', 'position_lat_id', 'position_lon_id']].copy()

df_powerplants['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_powerplants['cloud_fraction']))

df_powerplants['log_night_avg_rad'] = np.log(df_powerplants['night_avg_rad'])

df_powerplants.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)
X_powerplants = np.array(df_powerplants.drop(['NO2_column_number_density', 'position_lat_id', 'position_lon_id'], axis=1))

y_powerplants = np.array(df_powerplants['NO2_column_number_density'])

y_powerplants = y_powerplants*(10**5) #The numbers are very small so rescaling for model training and prediction

df_powerplants['predict_vehicles'] = model.predict(X_powerplants)*(10**-5)
df_powerplants['NO2_excess'] = df_powerplants['NO2_column_number_density'] - df_powerplants['predict_vehicles']



# In some cases, the predicted vehicle emissions were higher than the measured emissions in a grid cell. 

# The values in these cells were converted to zero in order to quantify the emissions factor

df_powerplants.loc[df_powerplants['NO2_excess']<0, 'NO2_excess'] = 0
#mol/m^2 per day converted to mol/year, a grid cell is approximately 1 km^2

total_emissions = round(df_powerplants['NO2_excess'].sum()*(1000*1000)*365, 2) 

print(total_emissions)
#gwh for a year using all sources of power including wind, hydro, and solar

total_generated = round(plants['estimated_generation_gwh'].sum())

print(total_generated)
#(mol NO2/year)/(mwh/year) = mol NO2/mwh of fossil fuel generated

emissions_factor = total_emissions/total_generated #(mol NO2/year)/(mwh/year) = mol NO2/mwh of fossil fuel generated

print(round(emissions_factor, 3))
# Read in data from the saved file created using the code in the regridding section above.

ds_monthly = xr.open_dataset('../input/ds4g-emissions-nc/monthly_ds4g_emissions.nc')
ds_monthly['landcover_category'] = ds_monthly['landcover_category'].fillna(0)

ds_monthly_vehicles = ds_monthly.where((ds_monthly.land_mask == 1) & 

                   (ds_monthly.no_plants_mask == 1) & 

                   (ds_monthly.landcover_category == 0))

ds_monthly_vehicles = ds_monthly_vehicles.drop(['no_plants_mask','plants_mask','land_mask', 'position_lat_id', 'position_lon_id'])

df_monthly_vehicles = ds_monthly_vehicles.to_dataframe().dropna()

df_monthly_vehicles = df_monthly_vehicles[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean']].copy()

df_monthly_vehicles['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_monthly_vehicles['cloud_fraction']))

df_monthly_vehicles['log_night_avg_rad'] = np.log(df_monthly_vehicles['night_avg_rad'])

df_monthly_vehicles.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)
df_monthly_vehicles2 = df_monthly_vehicles.copy()

df_monthly_vehicles2['NO2_column_number_density'] = df_monthly_vehicles['NO2_column_number_density']*(10**5)

sns.pairplot(df_monthly_vehicles2)
X_monthly_vehicles = np.array(df_monthly_vehicles.drop(['NO2_column_number_density'], axis=1))

y_monthly_vehicles = np.array(df_monthly_vehicles['NO2_column_number_density'])

y_monthly_vehicles = y_monthly_vehicles*(10**5) #The numbers are very small so rescaling for model training and prediction

X_monthly_train, X_monthly_test, y_monthly_train, y_monthly_test = train_test_split(X_monthly_vehicles, y_monthly_vehicles, test_size=0.33, random_state=122)

model_monthly = XGBRegressor(learning_rate = 0.01, objective='reg:squarederror', n_estimators = 2000)

model_monthly.fit(X_monthly_train, y_monthly_train)

y_monthly_train_pred = model_monthly.predict(X_monthly_train)

y_monthly_pred = model_monthly.predict(X_monthly_test)
plt.scatter(y_monthly_test, y_monthly_pred)  #The numbers are very small, multiply numbers in axes by 10**-5 for actual values
print(r2_score(y_monthly_train, y_monthly_train_pred))

print(r2_score(y_monthly_test, y_monthly_pred))

print(mean_squared_error(y_monthly_test*10**-5, y_monthly_pred*10**-5))

print(mean_absolute_error(y_monthly_test*10**-5, y_monthly_pred*10**-5))
ds_monthly_powerplants = ds_monthly.where(ds_monthly.plants_mask == 1) 

ds_monthly_powerplants = ds_monthly_powerplants.drop(['no_plants_mask','plants_mask','land_mask'])

df_monthly_powerplants = ds_monthly_powerplants.to_dataframe().dropna()

df_monthly_powerplants = df_monthly_powerplants[['NO2_column_number_density', 'cloud_fraction', 'night_avg_rad', 'gldas_wind_mean', 'gldas_airT_max', 'gldas_pres_mean', 'gldas_lwdown_mean']].copy()

df_monthly_powerplants['arcsin_cloud_fraction'] = np.arcsin(np.sqrt(df_monthly_powerplants['cloud_fraction']))

df_monthly_powerplants['log_night_avg_rad'] = np.log(df_monthly_powerplants['night_avg_rad'])

df_monthly_powerplants.drop(['cloud_fraction', 'night_avg_rad'], axis=1, inplace=True)
X_monthly_powerplants = np.array(df_monthly_powerplants.drop(['NO2_column_number_density'], axis=1))

y_monthly_powerplants = np.array(df_monthly_powerplants['NO2_column_number_density'])

y_monthly_powerplants = y_monthly_powerplants*(10**5) #The numbers are very small so rescaling for model training and prediction

df_monthly_powerplants['predict_vehicles'] = model_monthly.predict(X_monthly_powerplants)*(10**-5)
df_monthly_powerplants['NO2_excess'] = df_monthly_powerplants['NO2_column_number_density'] - df_monthly_powerplants['predict_vehicles']

df_monthly_powerplants[df_monthly_powerplants['NO2_excess']<0] = 0
monthly_EF = pd.DataFrame(df_monthly_powerplants['NO2_excess'].groupby('time').sum())

monthly_EF['date'] = monthly_EF.index

monthly_EF['days_in_month'] = monthly_EF['date'].dt.day

monthly_EF.drop('date', axis=1, inplace=True)
#mol/m^2 per day converted to mol/month

monthly_EF['total_emissions'] = round(monthly_EF['NO2_excess']*(1000*1000)*monthly_EF['days_in_month'], 2) 

monthly_EF['total_generated'] = round(plants['estimated_generation_gwh'].sum()/12)

monthly_EF['EF'] = monthly_EF['total_emissions']/monthly_EF['total_generated']
monthly_EF
plt = monthly_EF.EF.plot()
plants_fossil = pd.read_csv('../input/ds4g-emissions-nc/plants_fossil.csv')

plants_fossil_sum = plants_fossil[['position_lat', 'position_lon', 'estimated_generation_gwh', 'num_plants']].groupby(['position_lat', 'position_lon'], as_index=False).sum()

plants_fossil_1plant = pd.merge(plants_fossil, plants_fossil_sum[plants_fossil_sum['num_plants']==1])

plants_fossil_1plant = plants_fossil_1plant[['position_lat', 'position_lon', 'primary_fuel']]

df_powerplants_em = df_powerplants[['position_lat_id', 'position_lon_id', 'predict_vehicles', 'NO2_excess']].copy()

df_powerplants_loc = df_powerplants_em.groupby(['position_lat_id', 'position_lon_id'], as_index=False).sum()

df_powerplants_loc[['position_lat_id','position_lon_id']] = df_powerplants_loc[['position_lat_id','position_lon_id']].applymap(np.int64)

df_powerplants_loc.rename({'position_lat_id':'position_lat', 'position_lon_id':'position_lon'}, axis=1, inplace=True)

plants_1type = pd.merge(plants_fossil_sum, df_powerplants_loc, how='left', on=['position_lat','position_lon'])

plants_alltypes = pd.merge(plants_1type, plants_fossil_1plant, how='left', on=['position_lat','position_lon'])

plants_alltypes['primary_fuel'] = plants_alltypes['primary_fuel'].fillna('Mixed')

plants_primaryfuel = plants_alltypes[['primary_fuel','NO2_excess', 'estimated_generation_gwh', 'num_plants']].groupby(['primary_fuel'], as_index=False).sum()

plants_primaryfuel['NO2_excess_annual'] = plants_primaryfuel['NO2_excess']*(1000*1000)*365

plants_primaryfuel['EF'] = plants_primaryfuel['NO2_excess_annual']/plants_primaryfuel['estimated_generation_gwh']
plants_primaryfuel