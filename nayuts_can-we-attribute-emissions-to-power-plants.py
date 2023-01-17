from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

import folium

import math

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import rasterio as rio

import seaborn as sns

from sklearn.cluster import KMeans

import tifffile as tiff 
def overlay_image_on_puerto_rico_df(df, img, zoom):

    lat_map=df.iloc[[0]].loc[:,["latitude"]].iat[0,0]

    lon_map=df.iloc[[0]].loc[:,["longitude"]].iat[0,0]

    m = folium.Map([lat_map, lon_map], zoom_start=zoom)

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    folium.raster_layers.ImageOverlay(

        image=img,

        bounds = [[18.56,-67.32,],[17.90,-65.194]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df["latitude"].iloc[i],df["longitude"].iloc[i]],

                     icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(m)

        

    return m
def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)

power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)

power_plants['latitude'] = power_plants['latitude'].astype(float)

a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8

power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 

power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()
power_plants_df.head()
#From https://www.kaggle.com/ajulian/capacity-factor-in-power-plants



total_capacity_mw = power_plants_df['capacity_mw'].sum()

print('Total Installed Capacity: '+'{:.2f}'.format(total_capacity_mw) + ' MW')

capacity = (power_plants_df.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()

capacity = capacity.sort_values('capacity_mw',ascending=False)

capacity['percentage_of_total'] = (capacity['capacity_mw']/total_capacity_mw)*100

capacity.sort_values(by='percentage_of_total', ascending=True)['percentage_of_total'].plot(kind='bar',color=['lightblue', 'green', 'orange', 'black','lightgray','darkblue'])

image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')

overlay_image_on_puerto_rico_df(power_plants_df,image[:,:,0],8)



#https://www.kaggle.com/paultimothymooney/explore-image-metadata-s5p-gfs-gldas

#band1: NO2_column_number_density
lon = []

lat = []

NO2 = []



for i in range(image[:,:,0].shape[0]):

    for j in range(image[:,:,0].shape[1]):

        #print(image[:,:,0][i,j])

        NO2.append(image[:,:,0][i,j])

        lon.append(i)

        lat.append(j)

        

NO2 = np.array(NO2)

lon = np.array(lon)

lat = np.array(lat)
results = pd.DataFrame(columns=['NO2', 'lat', 'lon'])

results = pd.DataFrame({'NO2': NO2/max(NO2),

                    'lat': lat/max(lat),

                    'lon': lon/max(lon)})
sns.distplot(results["NO2"])
pred = KMeans(n_clusters=11).fit_predict(results)
plt.figure()

sns.heatmap(pred.reshape((148, 475)))
overlay_image_on_puerto_rico_df(power_plants_df, pred.reshape((148, 475)), 8)
monotonous = KMeans(n_clusters=3).fit_predict(image[:,:,0].reshape(-1, 1))

#pred = KMeans(n_clusters=2).fit_predict(results["NO2"])

plt.figure()

sns.heatmap(monotonous.reshape((148, 475)))
#Note that the color intensity and the high NO2 concentration do not always match.

overlay_image_on_puerto_rico_df(power_plants_df, monotonous.reshape((148, 475)), 8)
import os 

import glob
gldas_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/*')

gldas_files = sorted(gldas_files)

gfs_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/*')

gfs_files = sorted(gfs_files)
gldas_files_par_day = []

for i in range(0,len(gldas_files[6:54]),8):

    #print(gldas_files[i:i+8])

    gldas_files_par_day.append(gldas_files[i:i+8])
gfs_files_par_day = []

for i in range(0,len(gfs_files[3:27]),4):

    #print(gfs_files[i:i+4])

    gfs_files_par_day.append(gfs_files[i:i+4])
image_reglession_u = []

image_reglession_v = []

image_reglession_speed = []



for i in range(len(gfs_files_par_day)):

    gfs_tmp = gfs_files_par_day[i]

    gldas_tmp = gldas_files_par_day[i]

    array_wind_u = []

    array_wind_v = []

    array_wind_speed = []

    for j in range(len(gfs_tmp)):

        gfs_image_u = tiff.imread(gfs_tmp[j])[:,:,3]

        gfs_image_v = tiff.imread(gfs_tmp[j])[:,:,4]

        gldas_image1 = tiff.imread(gldas_tmp[2*j])[:,:,11]

        gldas_image2 = tiff.imread(gldas_tmp[2*j + 1])[:,:,11]



        #fill na by mean

        gfs_image_u = np.nan_to_num(gfs_image_u, nan=np.nanmean(gfs_image_u))

        gfs_image_v = np.nan_to_num(gfs_image_v, nan=np.nanmean(gfs_image_v))

        gldas_image1 = np.nan_to_num(gldas_image1, nan=np.nanmean(gldas_image1))

        gldas_image2 = np.nan_to_num(gldas_image2, nan=np.nanmean(gldas_image2))

        

        

        gldas_image = (gldas_image1 + gldas_image2)/2

        

        array_wind_u.append(gfs_image_u)

        array_wind_v.append(gfs_image_v)

        array_wind_speed.append(gldas_image)

        

        image_reglession_u.append(np.nanmean(np.array(array_wind_u), axis=0))

        image_reglession_v.append(np.nanmean(np.array(array_wind_v), axis=0))

        image_reglession_speed.append(np.nanmean(np.array(array_wind_speed), axis=0))

       

image_reglession_u = np.nanmean(np.array(image_reglession_u), axis=0)

image_reglession_v = np.nanmean(np.array(image_reglession_v), axis=0)

image_reglession_speed = np.nanmean(np.array(image_reglession_speed), axis=0)
sns.heatmap(image_reglession_u.reshape((148, 475)))
sns.heatmap(image_reglession_v.reshape((148, 475)))
sns.heatmap(image_reglession_speed.reshape((148, 475)))
image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')

lon = []

lat = []

NO2 = []

wind_u = []

wind_v = []

wind_speed = []



for i in range(image[:,:,0].shape[0]):

    for j in range(image[:,:,0].shape[1]):

        #print(image[:,:,0][i,j])

        NO2.append(image[:,:,0][i,j])

        lon.append(i)

        lat.append(j)

        wind_u.append(image_reglession_u.reshape((148, 475))[i,j])

        wind_v.append(image_reglession_v.reshape((148, 475))[i,j])

        wind_speed.append(image_reglession_speed.reshape((148, 475))[i,j])

        

NO2 = np.array(NO2)

lon = np.array(lon)

lat = np.array(lat)

wind_u = np.array(wind_u)

wind_v = np.array(wind_v)

wind_spped = np.array(wind_speed)

        

results_wind = pd.DataFrame(columns=['NO2', 'lat', 'lon', 'wind_u', 'wind_v', 'wind_speed'])

results_wind = pd.DataFrame({

                    'NO2': NO2/max(NO2),

                    'lat': lat/max(lat),

                    'lon': lon/max(lon),

                    'wind_u' : wind_u/(- min(wind_u)),

                    'wind_v' : wind_v/(- min(wind_v)),

                    'wind_speed': wind_speed/max(wind_speed)})
pred_wind1 = KMeans(n_clusters=11).fit_predict(results_wind)

plt.figure()

sns.heatmap(pred_wind1.reshape((148, 475)))
overlay_image_on_puerto_rico_df(power_plants_df, pred_wind1.reshape((148, 475)), 8)
image_reglession_u = []

image_reglession_v = []



for i in range(len(gfs_files_par_day)):

    gfs_tmp = gfs_files_par_day[i]

    gldas_tmp = gldas_files_par_day[i]

    array_wind_u = []

    array_wind_v = []

    for j in range(len(gfs_tmp)):

        gfs_image_u = tiff.imread(gfs_tmp[j])[:,:,3]

        gfs_image_v = tiff.imread(gfs_tmp[j])[:,:,4]

        gldas_image1 = tiff.imread(gldas_tmp[2*j])[:,:,11]

        gldas_image2 = tiff.imread(gldas_tmp[2*j + 1])[:,:,11]



        #fill na by mean

        gfs_image_u = np.nan_to_num(gfs_image_u, nan=np.nanmean(gfs_image_u))

        gfs_image_v = np.nan_to_num(gfs_image_v, nan=np.nanmean(gfs_image_v))

        gldas_image1 = np.nan_to_num(gldas_image1, nan=np.nanmean(gldas_image1))

        gldas_image2 = np.nan_to_num(gldas_image2, nan=np.nanmean(gldas_image2))

        

        

        gldas_image = (gldas_image1 + gldas_image2)/2

        wind_u = gfs_image_u * gldas_image

        wind_v = gfs_image_v * gldas_image

        

        array_wind_u.append(wind_u)

        array_wind_v.append(wind_v)

        

        image_reglession_u.append(np.nanmean(np.array(array_wind_u), axis=0))

        image_reglession_v.append(np.nanmean(np.array(array_wind_v), axis=0))

       

image_reglession_u = np.nanmean(np.array(image_reglession_u), axis=0)

image_reglession_v = np.nanmean(np.array(image_reglession_v), axis=0)
sns.heatmap(image_reglession_u.reshape((148, 475)))
sns.heatmap(image_reglession_v.reshape((148, 475)))
image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')

lon = []

lat = []

NO2 = []

wind_u = []

wind_v = []



for i in range(image[:,:,0].shape[0]):

    for j in range(image[:,:,0].shape[1]):

        #print(image[:,:,0][i,j])

        NO2.append(image[:,:,0][i,j])

        lon.append(i)

        lat.append(j)

        wind_u.append(image_reglession_u.reshape((148, 475))[i,j])

        wind_v.append(image_reglession_v.reshape((148, 475))[i,j])

        

NO2 = np.array(NO2)

lon = np.array(lon)

lat = np.array(lat)

wind_u = np.array(wind_u)

wind_v = np.array(wind_v)

        

results_wind = pd.DataFrame(columns=['NO2', 'lat', 'lon', 'wind_u', 'wind_v'])

results_wind = pd.DataFrame({

                    'NO2': NO2/max(NO2),

                    'lat': lat/max(lat),

                    'lon': lon/max(lon),

                    'wind_u' : wind_u/(- min(wind_u)),

                    'wind_v' : wind_v/(- min(wind_v))})



pred_wind2 = KMeans(n_clusters=11).fit_predict(results_wind)

plt.figure()

sns.heatmap(pred_wind2.reshape((148, 475)))
overlay_image_on_puerto_rico_df(power_plants_df, pred_wind2.reshape((148, 475)), 8)
plt.figure()

sns.heatmap(pred.reshape((148, 475)))
plt.figure()

sns.heatmap(pred_wind1.reshape((148, 475)))
plt.figure()

sns.heatmap(pred_wind2.reshape((148, 475)))