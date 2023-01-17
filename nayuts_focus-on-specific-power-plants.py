from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

import folium

import math

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import rasterio as rio

import seaborn as sns

import tifffile as tiff 
def overlay_image_on_puerto_rico_with_Marker(df,file_name,band_layer,lat,lon,zoom):

    """

    Visualize map overlayed data & plant markers.

    """

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=zoom)

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df["latitude"].iloc[i],df["longitude"].iloc[i]],

                     icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(m)

        

    return m
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
def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup,

                     icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(plot)

    return(plot)
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
power_plants_df.columns
power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']].head()
image ='/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif'

latitude=18.1429005246921; longitude=-65.4440010699994

overlay_image_on_puerto_rico_with_Marker(power_plants_df,image,band_layer=1,lat=latitude,lon=longitude,zoom=8)



#https://www.kaggle.com/paultimothymooney/explore-image-metadata-s5p-gfs-gldas

#band1: NO2_column_number_density

#band2: tropospheric_NO2_column_number_density

#band3: stratospheric_NO2_column_number_density

#band4: NO2_slant_column_number_density
img = tiff.imread(image)[:,:,0]

print("mean: ", img.mean())

print("max: ", img.max())

print("min: ", img.min())
sns.distplot(img.reshape([70300,]), kde=False, rug=True)
power_plant_name = "Costa Sur"

power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]

power_plants_specific
lat_map=power_plants_specific.iloc[[0]].loc[:,["latitude"]].iat[0,0]

lon_map=power_plants_specific.iloc[[0]].loc[:,["longitude"]].iat[0,0]

plot_points_on_map(power_plants_df[power_plants_df["name"] == power_plant_name],0,425,'latitude',lat_map,'longitude',lon_map,13)
MASK_SIZE = 10
image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')
def create_mask(img, df, MASK_SIZE):

    """

    input: 

       img: orginal image 

       df: dataframe of focused power plants

       MASK_SIZE

    output:

       numpyarray same size as img.

    ----------

    ・1st: Create zero matrix which size is same as the whole data.

    ・2nd： Create implicit mask ( this express "region around the focused plant") as a two-dimensional range for each power plants.

    ・3nd: Replace elements with orginal data if the element in mask.

    """

    

    mask = np.zeros((img.shape[0], img.shape[1]))

    values_in_mask = np.array([])

    

    for i in range(len(df)):

        lat = float(df.iloc[[i]].loc[:,["latitude"]].iat[0,0])

        lon = float(df.iloc[[i]].loc[:,["longitude"]].iat[0,0])



    

        f_lat = (lat - 17.903121359128956)*img.shape[0]/(18.563112930177304 - 17.903121359128956)

        f_lon = (lon + 67.32297404549217)*img.shape[1]/(-65.19437297127342 + 67.32297404549217)

        f_lat_int = int(Decimal(str(f_lat -1)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        f_lon_int = int(Decimal(str(f_lon -1)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        

        mask_lat_min = 148  - f_lat_int - MASK_SIZE

        mask_lat_max = 148  - f_lat_int + MASK_SIZE

        mask_lon_min = f_lon_int - MASK_SIZE 

        mask_lon_max = f_lon_int + MASK_SIZE 

        #print(mask_lat_min, mask_lat_max, mask_lon_min, mask_lon_max, f_lat, f_lon)

        

        for i in range(img.shape[0]):

            for j in range(img.shape[1]):

                if math.sqrt( (i- (147 - f_lat_int))**2 + (j  - f_lon_int )**2) <= MASK_SIZE:

                    

                    if image[:,:,0][i][j] < 0:

                        continue

                        

                    mask[i,j] = image[:,:,0][i][j]

                    values_in_mask = np.append(values_in_mask, mask[i,j])

                #if i in range(mask_lat_min,mask_lat_max) and j in range(mask_lon_min,mask_lon_max):

                #    mask[i,j] = image[:,:,0][i][j]

                #    values_in_mask = np.append(values_in_mask, mask[i,j])



    #print(len(values_in_mask))

    print(power_plant_name, " is:")

    sns.distplot(values_in_mask[:400], kde=False, rug=True)



    return mask
power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]
power_plants_specific = power_plants_specific.reset_index(drop=True)
mask = create_mask(image, power_plants_specific, MASK_SIZE)
overlay_image_on_puerto_rico_df(power_plants_specific, mask,zoom=12)
print("mean: ", mask.mean())

print("max: ", mask.max())

print("min: ", mask.min())
power_plant_name = "San Juan CC"

power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]

power_plants_specific = power_plants_specific.reset_index(drop=True)

mask = create_mask(image, power_plants_df[power_plants_df["name"] == power_plant_name], MASK_SIZE)

overlay_image_on_puerto_rico_df(power_plants_df[power_plants_df["name"] == power_plant_name], mask,zoom=12)
print("mean: ", mask.mean())

print("max: ", mask.max())

print("min: ", mask.min())
power_plant_name = "Palo Seco"

power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]

power_plants_specific = power_plants_specific.reset_index(drop=True)

mask = create_mask(image, power_plants_df[power_plants_df["name"] == power_plant_name], MASK_SIZE)

overlay_image_on_puerto_rico_df(power_plants_specific, mask, zoom=12)
print("mean: ", mask.mean())

print("max: ", mask.max())

print("min: ", mask.min())
import glob

import re

import os

from datetime import datetime, timedelta



s5p_no2_timeseries = glob.glob('../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/*')

print("There are ", len(s5p_no2_timeseries), "s5p_no2 time series data.")

print("  --------------------  ")

print(s5p_no2_timeseries)

dates = [datetime.strptime(i[:79], '../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_%Y%m%d') for i in  s5p_no2_timeseries]
#just stopped plotting graph version.

def create_mask_without_plot(img, df, MASK_SIZE):

    """

    input: 

       img: orginal image 

       df: dataframe of focused power plants

       MASK_SIZE

    output:

       numpyarray same size as img.

    ----------

    ・1st: Create zero matrix which size is same as the whole data.

    ・2nd： Create implicit mask ( this express "region around the focused plant") as a two-dimensional range for each power plants.

    ・3nd: Replace elements with orginal data if the element in mask.

    """

    

    mask = np.zeros((img.shape[0], img.shape[1]))

    values_in_mask = np.array([])

    

    for i in range(len(df)):

        lat = float(df.iloc[[i]].loc[:,["latitude"]].iat[0,0])

        lon = float(df.iloc[[i]].loc[:,["longitude"]].iat[0,0])



    

        f_lat = (lat - 17.903121359128956)*img.shape[0]/(18.563112930177304 - 17.903121359128956)

        f_lon = (lon + 67.32297404549217)*img.shape[1]/(-65.19437297127342 + 67.32297404549217)

        f_lat_int = int(Decimal(str(f_lat -1)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        f_lon_int = int(Decimal(str(f_lon -1)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))

        

        mask_lat_min = 148  - f_lat_int - MASK_SIZE

        mask_lat_max = 148  - f_lat_int + MASK_SIZE

        mask_lon_min = f_lon_int - MASK_SIZE 

        mask_lon_max = f_lon_int + MASK_SIZE 

        #print(mask_lat_min, mask_lat_max, mask_lon_min, mask_lon_max, f_lat, f_lon)

        

        for i in range(img.shape[0]):

            for j in range(img.shape[1]):

                if math.sqrt( (i- (147 - f_lat_int))**2 + (j  - f_lon_int )**2) <= MASK_SIZE:

                    

                    if image[:,:,0][i][j] < 0:

                        continue



                    mask[i,j] = image[:,:,0][i][j]

                    values_in_mask = np.append(values_in_mask, mask[i,j])

                #if i in range(mask_lat_min,mask_lat_max) and j in range(mask_lon_min,mask_lon_max):

                #    mask[i,j] = image[:,:,0][i][j]

                #    values_in_mask = np.append(values_in_mask, mask[i,j])



    #print(len(values_in_mask))

    #print(power_plant_name, " is:")

    #sns.distplot(values_in_mask[:400], kde=False, rug=True)



    return mask
stat_max = np.array([])

stat_mean = np.array([])

stat_min = np.array([])



for name in set(power_plants_df["name"]):

    power_plant_name = name

    power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]

    power_plants_specific = power_plants_specific.reset_index(drop=True)



    stat_max_tmp = []

    stat_mean_tmp = []

    stat_min_tmp = []



    for i in s5p_no2_timeseries:  

        image = tiff.imread(i)

        mask = create_mask_without_plot(image, power_plants_df[power_plants_df["name"] == power_plant_name], MASK_SIZE)

        stat_max_tmp.append(mask.max())

        stat_mean_tmp.append(mask.mean())

        stat_min_tmp.append(mask.min())

        

    stat_max = np.append(stat_max, stat_max_tmp)

    stat_mean = np.append(stat_mean, stat_mean_tmp)

    stat_min = np.append(stat_min, stat_min_tmp)
stat_max_reshape = stat_max.reshape(32, int(len(stat_max)/32))

stat_mean_reshape = stat_mean.reshape(32, int(len(stat_mean)/32))

stat_min_reshape = stat_min.reshape(32,int(len(stat_min)/32))
stat_max_nanmean = np.nanmean(stat_max_reshape, axis=0)

stat_mean_nanmean = np.nanmean(stat_mean_reshape, axis=0)

stat_min_nanmean = np.nanmean(stat_min_reshape, axis=0)
#results = pd.DataFrame(index=dates, data=stat[:387], columns=['San Juan CC'])

results = pd.DataFrame(columns=['max', 'mean', 'min'])

results = pd.DataFrame({'max': stat_max_nanmean,

                    'mean': stat_mean_nanmean,

                    'min': stat_min_nanmean},

                    index=dates)
results.head()
results[['max','min']].plot()

plt.title('Max of NO2_column_number_density in Puerto Rico')
results[['mean','min']].plot()

plt.title('Mean of NO2_column_number_density in Puerto Rico')
#power_plant_name = "San Juan CC"

#power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]

#Quanity_of_electricity_generated = power_plants_specific.loc[:,["estimated_generation_gwh"]].iat[0,0]

Quanity_of_electricity_generated = np.sum(power_plants_df.loc[:,["estimated_generation_gwh"]])
(results[['mean','min']]/Quanity_of_electricity_generated[0]).plot()

plt.title('Mean Simplified Emissions Factor in Puerto Rico')

#plt.title('Mean Simplified Emissions Factor in San Juan CC')
Simplified_Emissions_Factor = (results.loc[:,['mean']]/Quanity_of_electricity_generated[0]).mean().at['mean']
print(f"Simplified Emissions Factor of {power_plant_name}:",Simplified_Emissions_Factor,"mol/gwh・m^2")
images_avg =  np.array([])

for i in s5p_no2_timeseries:

    image = tiff.imread(i)

    

    images_avg = np.append(np.nanmean(image[:,:,0]), images_avg)
masks_nonzero_avg = np.array([])



for name in set(power_plants_df["name"]):

    power_plant_name = name

    power_plants_specific = power_plants_df[power_plants_df["name"] == power_plant_name]

    power_plants_specific = power_plants_specific.reset_index(drop=True)





    for i in s5p_no2_timeseries:  

        image = tiff.imread(i)

        mask = create_mask_without_plot(image, power_plants_df[power_plants_df["name"] == power_plant_name], MASK_SIZE)

        mask_nonzero = [i for i in mask.reshape([70300,]) if i > 0]

        masks_nonzero_avg = np.append(masks_nonzero_avg, np.nanmean(mask_nonzero))
sns.distplot(images_avg, kde=False, rug=True)

print("s5p_no2 data mean:", np.nanmean(images_avg) )

print("s5p_no2 data variable:", np.nanvar(images_avg) )
sns.distplot(masks_nonzero_avg, kde=False, rug=True)

print("masked s5p_no2 data mean:", np.nanmean(masks_nonzero_avg) )

print("masked s5p_no2 data variable:", np.nanvar(masks_nonzero_avg) )