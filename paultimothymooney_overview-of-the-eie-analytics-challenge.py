import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import rasterio as rio

import folium

import tifffile as tiff 



        

def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup).add_to(plot)

    return(plot)



def overlay_image_on_puerto_rico(file_name,band_layer,lat,lon,zoom):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=zoom)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

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

lat=18.200178; lon=-66.664513

plot_points_on_map(power_plants,0,425,'latitude',lat,'longitude',lon,9)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif'

latitude=18.1429005246921; longitude=-65.4440010699994

overlay_image_on_puerto_rico(image,band_layer=1,lat=latitude,lon=longitude,zoom=8)
lat=18.1429005246921; lon=-65.4440010699994

plot_points_on_map(power_plants,0,425,'latitude',lat,'longitude',lon,12)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif'

latitude =18.1429005246921; longitude =-65.4440010699994

overlay_image_on_puerto_rico(image,band_layer=1,lat=latitude,lon=longitude,zoom=12)
power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()

power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']][29:30]

quantity_of_electricity_generated = power_plants_df['estimated_generation_gwh'][29:30].values

print('Quanity of Electricity Generated: ', quantity_of_electricity_generated)
# This is just an example to illustrate that you can extract numerical values from .tiff files

# Ideally you would limit to only the bands that are related to NO2 emissions

# Likewise you might want to limit the data to only the region of interest

average_no2_emission = [np.average(tiff.imread(image))]

print('Average NO2 emissions value: ', average_no2_emission)
simplified_emissions_factor = float(average_no2_emission/quantity_of_electricity_generated)

print('Simplified emissions factor (S.E.F.) for a single power plant on the island of Vieques =  \n\n', simplified_emissions_factor, 'S.E.F. units')