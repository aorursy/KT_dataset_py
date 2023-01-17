# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import rasterio as rio

import folium
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
power_plants=pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)

power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)

power_plants['latitude'] = power_plants['latitude'].astype(float)

a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8

power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 

lat=18.200178; lon=-66.664513

plot_points_on_map(power_plants,0,425,'latitude',lat,'longitude',lon,9)
power_plants_df =power_plants.sort_values('capacity_mw',ascending=False).reset_index()

power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']]
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180702_1500.tif'

image_band = rio.open(image).read(3)

plot_scaled(image_band)



image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018072118.tif'

image_band = rio.open(image).read(3)

plot_scaled(image_band)



overlay_image_on_puerto_rico(image,band_layer=3)
from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials

import ee

import folium
def add_ee_layer(self, ee_image_object, vis_params, name):

  # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)

  folium.raster_layers.TileLayer(

    tiles = map_id_dict['tile_fetcher'].url_format,

    attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',

    name = name,

    overlay = True,

    control = True

  ).add_to(self)
def plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom):

    # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

    folium.Map.add_ee_layer = add_ee_layer

    vis_params = {

      'min': minimum_value,

      'max': maximum_value,

      'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

    my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)

    s5p = ee.ImageCollection(dataset).filterDate(

        begin_date, end_date)

    my_map.add_ee_layer(s5p.first().select(column), vis_params, 'Color')

    my_map.add_child(folium.LayerControl())

    display(my_map)

ee.Authenticate()
!cat ~/.config/earthengine/credentials
user_secret = "earth_engine" # Your user secret, defined in the add-on menu of the notebook editor

refresh_token = '1//0eD5j0K8ku_huCgYIARAAGA4SNwF-L9Ir9BaJSkVwzCcscdLpMSgDtqx5nDEb7fNiZJywieYPHKxh1j0H5dZu_ahn8PerVQqf4SU'

credentials = Credentials(

        None,

        refresh_token=refresh_token,

        token_uri=ee.oauth.TOKEN_URI,

        client_id=ee.oauth.CLIENT_ID,

        client_secret=ee.oauth.CLIENT_SECRET,

        scopes=ee.oauth.SCOPES)

ee.Initialize(credentials=credentials)
dataset = "COPERNICUS/S5P/NRTI/L3_NO2"

column = 'absorbing_aerosol_index'

begin_date = '2018-07-08'

end_date = '2018-07-14'

minimum_value = 0.0000000001

maximum_value = 1

latitude = 60.17

longitude = 25.94

zoom = 5

plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)
dataset = "NOAA/GFS0P25"

column = 'temperature_2m_above_ground'

begin_date = '2018-07-08'

end_date = '2018-07-14'

minimum_value = 0

maximum_value = 50

latitude = 18.20

longitude = -66.66

zoom = 1

plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)



dataset = "NASA/GLDAS/V021/NOAH/G025/T3H"

column = 'Tair_f_inst'

begin_date = '2018-07-08'

end_date = '2018-07-14'

minimum_value = 270

maximum_value = 310

latitude = 18.20

longitude = -66.66

zoom = 6

plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)