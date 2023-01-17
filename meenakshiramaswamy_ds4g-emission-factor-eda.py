import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import rasterio as rio

import folium 

import seaborn as sns


def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):

    df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    plot = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,len(df)):

        popup = folium.Popup(str(df.primary_fuel[i:i+1]))

        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup).add_to(plot)

    return(plot)



def overlay_image_on_puerto_rico(file_name,band_layer):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=8)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    return m



def plot_scaled(file_name):

    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch

    img_plt = plt.imshow(file_name, cmap='gray', vmin=vmin, vmax=vmax)

    plt.show()



def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
pd.set_option('max_columns', 30)
power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')



power_plants.head(35)
power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)

power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)

power_plants['latitude'] = power_plants['latitude'].astype(float)

a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8

power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 

lat=18.200178; lon=-66.664513

plot_points_on_map(power_plants,0,425,'latitude',lat,'longitude',lon,9)
power_plants.columns
import pandas_profiling

eda_analysis = pandas_profiling.ProfileReport(power_plants)

eda_analysis.to_file('eie_analysis.html')
power_plants.info()
power_plants.describe(include = 'all')
#power_plants_df = power_plants.sort_values('estimated_generation_gwh',ascending=False).reset_index()

dsg_df = power_plants[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh','source','owner','country','commissioning_year','year_of_capacity_data']]
dsg_df.head(25)
dsg_df.describe()
plt.figure(figsize=(25,15))

sns.barplot(x='capacity_mw', y='estimated_generation_gwh', hue='primary_fuel', data=dsg_df[dsg_df['primary_fuel'].isin(['Coal','Oil','Gas'])])

#plt.set_xticklabels(a.get_xticklabels(), rotation=45)

plt.ylabel('Estimated Generation')

plt.title('Capacity Vs Estimated Generation');
dsg_df_corr = dsg_df.corr()

dsg_df_corr = sns.heatmap(dsg_df_corr, cmap="Accent",  annot= True)
from matplotlib import style

from matplotlib.pyplot import pie,show
power_plants_df_fueltype = power_plants.groupby (['primary_fuel']).agg({'capacity_mw': 'sum',

                                                'estimated_generation_gwh': 'sum'           

                                                }).reset_index()
power_plants_df_fueltype[['primary_fuel','capacity_mw','estimated_generation_gwh']]
plt.figure(figsize=(22,15))

plt.title("Fuel Generation from different power plants")

colors = ['green', 'orange', 'pink', 'c', 'm', 'y']

power_plants['primary_fuel'].value_counts().plot(kind='pie', colors=colors, 

 autopct='%1.1f%%',

counterclock=False, shadow=True)

power_plants['primary_fuel'].value_counts()
%matplotlib inline 



fig, ax = plt.subplots(figsize=[8,6])

plt.title ('Various Power Plants Capacity (in MW)')

plt.xlabel('Fuel Type')

plt.ylabel('Capacity mw');

ax.bar(power_plants_df_fueltype['primary_fuel'], power_plants_df_fueltype['capacity_mw'], color='YGR')
import seaborn as sns

sns.jointplot(x="capacity_mw", y="estimated_generation_gwh", data=power_plants_df_fueltype, kind = 'kde', color = 'orange')
df = power_plants[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']]

d=df.corr()

plt.figure(figsize=(10,7))

a = sns.heatmap(d, cmap="viridis",  annot= True)

a.Title = 'Fuel Type power Generation - Heatmap';

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=45)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=45)

plt.show()
from skimage.io import imread

image = imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif')

print (image.shape)

plt.imshow(image[:,:,0], cmap = 'cool')

plt.axes = False
image = imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018070400.tif')

print (image.shape)

plt.imshow(image[:,:,2], cmap = 'viridis')
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180714T170945_20180720T185244.tif'

image_band = rio.open(image).read(1)

plot_scaled(image_band)

overlay_image_on_puerto_rico(image,band_layer=1)
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180702_1500.tif'

image_band = rio.open(image).read(3)

plot_scaled(image_band)



image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018072118.tif'

image_band = rio.open(image).read(3)

plot_scaled(image_band)



overlay_image_on_puerto_rico(image,band_layer=3)