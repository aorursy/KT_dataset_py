import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math as m



from datetime import datetime



import os

import glob



import rasterio as rio

import folium 



import geopandas

import tifffile as tiff

from folium import plugins

from shapely.geometry import Point

import rasterstats

from rasterstats import zonal_stats, point_query
pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)

import warnings

warnings.filterwarnings('ignore')

from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
def plot_points_on_map(dataframe,latitude_value,longitude_value,zoom):

    #df = dataframe[begin_index:end_index]

    location = [latitude_value,longitude_value]

    fuelColor = {'Coal': 'darkred',

                 'Oil': 'black',

                 'Gas': 'lightgray',

                 'Hydro': 'lightblue',

                 'Solar': 'orange',

                 'Wind': 'green'

                }

    Map = folium.Map(location=location,zoom_start=zoom)

    for i in range(0,dataframe.shape[0]):

        fuel = dataframe['primary_fuel'][i]

        lat = dataframe['latitude'][i]

        lon = dataframe['longitude'][i]

        name = dataframe['name'][i]

        #data = dataframe[i]

        popup = "{}, geolocation : [{},{}], {} ".format(fuel,lat, lon, name)

        #popup = pd.DataFrame(data).to_html()

        color = fuelColor[fuel]

        folium.Marker([lat,lon],

                      popup=popup,

                      icon=folium.Icon(color=color, icon_color='white', icon='bolt', prefix='fa')

                     ).add_to(Map)  

    



    return Map



    
def overlay_image_on_puerto_rico(file_name,band_layer,lat,lon,zoom):

    band = rio.open(file_name).read(band_layer)

    m = folium.Map([lat, lon], zoom_start=zoom)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds = [[18.6,-67.3,],[17.9,-65.2]],

        colormap=lambda x: (1, 0, 0, x),

    ).add_to(m)

    return m



def plot_scaled(file_name):

    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch

    img_plt = plt.imshow(file_name, cmap='terrain', vmin=vmin, vmax=vmax)

    plt.show()



def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):

    for i in range(0, len(dataframe)):

        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]

    return dataframe
import json



def string_to_dict(dict_string):

    # Convert to proper json format (from here: https://stackoverflow.com/questions/39169718/convert-string-to-dict-then-access-keyvalues-how-to-access-data-in-a-class)

    dict_string = dict_string.replace("'", '"').replace('u"', '"')

    return json.loads(dict_string)['coordinates']



#for kaggle

power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

data_path = '/kaggle/input/ds4g-environmental-insights-explorer'



# For colab

#data_path = '/content/drive/My Drive/Kaggle-DS4G'



power_plants = pd.read_csv(data_path+'/eie_data/gppd/gppd_120_pr.csv')



power_plants.head(5)

power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)

power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)

power_plants['latitude'] = power_plants['latitude'].astype(float)

a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8

power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 

power_plants['longitude'] = power_plants['longitude'].astype(float)

power_plants['coord'] = power_plants['.geo'].apply(string_to_dict)

#power_plants.head()
lat=18.200178; lon=-66.664513

plot_points_on_map(power_plants,lat,lon,10)
power_plants_df = power_plants[['capacity_mw', 

                               'commissioning_year', 

                               'country', 

                               'estimated_generation_gwh', 

                               'source', 'name', 

                               'owner', 'primary_fuel','latitude','longitude','wepp_id','coord']]
# for Colab

#data_path = '/content/drive/My Drive/Kaggle-DS4G'

#os.listdir(data_path + '/eie_data/gldas')

# for Kaggle

data_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data'
from skimage.io import imread

image = imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif')

print (image.shape)

plt.imshow(image[:,:,0], cmap = 'cool')

plt.axes = False
image = imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018070400.tif')

print (image.shape)

plt.imshow(image[:,:,2], cmap = 'pink')



image = imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180701_0600.tif')

print (image.shape)

plt.imshow(image[:,:,2], cmap = 'ocean')
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/gldas_20180702_1500.tif'

image_band = rio.open(image).read(3)

plot_scaled(image_band)



image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/gfs_2018072118.tif'

image_band = rio.open(image).read(3)

plot_scaled(image_band)



overlay_image_on_puerto_rico(image,band_layer=2, lat = lat, lon = lon, zoom = 9)
#@title



import ee

from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("akshi")

#@title



ee.Authenticate()

!cat ~/.config/earthengine/credentials

#@title



#user_secret = "" # Your user secret, defined in the add-on menu of the notebook editor

#refresh_token = UserSecretsClient().get_secret(user_secret)

credentials = Credentials(

        None,

        refresh_token=secret_value_0,

        token_uri=ee.oauth.TOKEN_URI,

        client_id=ee.oauth.CLIENT_ID,

        client_secret=ee.oauth.CLIENT_SECRET,

        scopes=ee.oauth.SCOPES)

ee.Initialize()

#ee.Initialize(credentials=credentials)

def initMap(df, lat, lon):

    location = [lat, lon]

    Map = folium.Map(location=location, zoom_start=8)

    return Map

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



folium.Map.add_ee_layer = add_ee_layer



def plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom):

    # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb

    folium.Map.add_ee_layer = add_ee_layer

    vis_params = {

      'min': minimum_value,

      'max': maximum_value,

      'palette': ','.join(['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026'])}

      #'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}

    my_map = folium.Map(location=[latitude,longitude],tiles="OpenStreetMap" , zoom_start=zoom, height=500)

    s5p = ee.ImageCollection(dataset).filterDate(

        begin_date, end_date)

    my_map.add_ee_layer(s5p.first().select(column), vis_params, 'Color')

    my_map.add_child(folium.LayerControl())

    display(my_map)
#Sentinel-5P OFFL NO2: Offline Nitrogen Dioxide

startdate = '2019-05-01'

finishdate = '2019-05-31'

latitude = 18.20

longitude = -66.66



ee_s5p_no2 = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')

              .select('tropospheric_NO2_column_number_density')

              .filterDate(startdate, finishdate)

             )

band_viz_s5p_no2 = {

    'min': 0,

    'max': 0.00015,

    'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}

Map = initMap(power_plants_df, lat = latitude, lon = longitude)



Map.add_ee_layer(ee_s5p_no2.mean(), band_viz_s5p_no2, 'S5P_NO2')

Map.add_child(folium.LayerControl())



Map
image = data_path + '/gfs/gfs_2018070306.tif'

im = rio.open(image)

im.descriptions

im.dataset_mask()

im.units
def preview_meta_data(file_name):

    with rio.open(file_name) as img_filename:

        print('Bounding Box:',img_filename.bounds)

        print('\nCoordinates of Top Left Corner: ',img_filename.transform * (0,0))

        print('\nCoordinates of Bottom Right Corner: ',img_filename.transform * (img_filename.width,img_filename.height))
preview_meta_data(image)
def plot_image_s5p(img, color):

  fig, ax = plt.subplots(3,4,figsize=(20,6))

  band=1

  image_s5p = rio.open(img)

  for n in range(3):

      for m in range(4):

          image_bandi = image_s5p.read(band)

          desc = image_s5p.descriptions[band-1]

          ax[n,m].set_title(desc)

          ax[n,m].imshow(image_bandi, cmap=color)

          ax[n,m].grid(False)

          band += 1

          

  fig.tight_layout()
image = data_path+'/s5p_no2/s5p_no2_20180728T160540_20180803T194754.tif'



latitude=18.1429005246921; longitude=-65.4440010699994

plot_image_s5p(image, 'inferno')

#overlay_image_on_puerto_rico(image,band_layer=2,lat=latitude,lon=longitude,zoom=8)

test_image_part1 = '/kaggle/input/geeimagesamples/20180712T174753_20180719T105625-0000000000-0000000000.tif'

plot_image_s5p(test_image_part1, 'inferno')

#overlay_image_on_puerto_rico(test_image,band_layer=7,lat=latitude,lon=longitude,zoom=9)
test_image_part2 =   '/kaggle/input/geeimagesamples/20180712T174753_20180719T105625-0000000000-0000006912.tif'

plot_image_s5p(test_image_part2, 'inferno')

imp_img = tiff.imread(test_image_part1)

#print (imp_img)

no2_mean_imp_img = np.nanmean(imp_img[:,:,0:4])

tropospheric_NO2_mean = np.nanmean(imp_img[:,:,1])

print ("tropospheric_NO2_mean : {:.8f}".format(tropospheric_NO2_mean))

print ("cloud fraction : {:.8f}".format(np.nanmean(imp_img[:,:,5])))

print ("aai : {:.8f}".format(np.nanmean(imp_img[:,:,6])))
test_image1_part1 =   '/kaggle/input/geeimagesamples1/20180707T160011_20180713T174831-0000000000-0000000000.tif'

plot_image_s5p(test_image1_part1, 'Oranges')
test_image1_part2 =  '/kaggle/input/geeimagesamples1/20180707T160011_20180713T174831-0000000000-0000006912.tif'

plot_image_s5p(test_image1_part2, 'Oranges')

imp_img = tiff.imread(test_image1_part2)

print (imp_img)
no2_mean_imp_img = np.nanmean(imp_img[:,:,0:4])

print ("no2_mean : {:.8f}".format(no2_mean_imp_img))

print ("cloud fraction : {:.8f}".format(np.nanmean(imp_img[:,:,5])))

print ("aai : {:.8f}".format(np.nanmean(imp_img[:,:,6])))
test_image2_part1 =  '/kaggle/input/geeimagesamples/20180712T160623_20180719T105633-0000000000-0000000000.tif'

plot_image_s5p(test_image2_part1, 'cubehelix')

imp_img = tiff.imread(test_image2_part1)

print (imp_img)

no2_mean_imp_img = np.nanmean(imp_img[:,:,0:4])

print ("no2_mean : {:.8f}".format(no2_mean_imp_img))

#print ("cloud fraction : {:.8f}".format(np.nanmean(imp_img[:,:,5])))

print ("aai : {:.8f}".format(np.nanmean(imp_img[:,:,6])))
test_image2_part2 =  '/kaggle/input/geeimagesamples/20180712T160623_20180719T105633-0000000000-0000006912.tif'

plot_image_s5p(test_image2_part2, 'cubehelix')

imp_img = tiff.imread(test_image2_part2)

print (imp_img)

no2_mean_imp_img = np.nanmean(imp_img[:,:,0:4])

tropospheric_NO2_mean = np.nanmean(imp_img[:,:,1])

print ("tropospheric_NO2_mean : {:.8f}".format(tropospheric_NO2_mean))

print ("no2_mean : {:.8f}".format(no2_mean_imp_img))

#print ("cloud fraction : {:.8f}".format(np.nanmean(imp_img[:,:,5])))

print ("aai : {:.8f}".format(np.nanmean(imp_img[:,:,6])))

preview_meta_data(test_image2_part2)
image = data_path+'/gfs/gfs_2018070306.tif'

im = rio.open(image)

im.descriptions

im.dataset_mask()

im.units

im.bounds
preview_meta_data(data_path+'/gfs/gfs_2018070306.tif')
latitude=18.1429005246921; longitude=-65.4440010699994

overlay_image_on_puerto_rico(image,band_layer=5,lat=latitude,lon=longitude,zoom=8)
def plot_image_dfs(img, color):

  fig, ax = plt.subplots(2,3,figsize=(20,6))

  band=1

  image_dfs = rio.open(img)

  for n in range(2):

      for m in range(3):

          image_bandi = image_dfs.read(band)

          desc = image_dfs.descriptions[band-1]

          ax[n,m].set_title(desc)

          ax[n,m].imshow(image_bandi, cmap=color)

          ax[n,m].grid(False)

          band += 1

          

  fig.tight_layout()
image = data_path+'/gfs/gfs_2018070306.tif'



latitude=18.1429005246921; longitude=-65.4440010699994

plot_image_dfs(image, 'gist_earth')
plot_image_dfs(image, 'viridis')

imp_img = tiff.imread(image)

print (imp_img)
u_wind_mean = np.nanmean(imp_img[:,:,3])

print ("u_wind : {:.8f}".format(u_wind_mean))

print ("v_wind : {:.8f}".format(np.nanmean(imp_img[:,:,4])))



#print ("percip : {:.8f}".format(np.nanmean(imp_img[:,:,5])))
image = data_path + '/gldas/gldas_20190528_0000.tif'

im = rio.open(image)

im.descriptions

im.count

im.tags(ns='IMAGE_STRUCTURE')
preview_meta_data(image)
def plot_image_gldas(img, color):

  fig, ax = plt.subplots(3,4,figsize=(20,6))

  band=1

  image_gldas = rio.open(img)

  for n in range(3):

      for m in range(4):

          image_bandi = image_gldas.read(band)

          desc = image_gldas.descriptions[band-1]

          ax[n,m].set_title(desc)

          ax[n,m].imshow(image_bandi, cmap=color)

          ax[n,m].grid(False)

          band += 1

          

  fig.tight_layout()
image = data_path + '/gldas/gldas_20190528_1200.tif'



plot_image_gldas(image, 'inferno')

imp_img = tiff.imread(image)

#print (imp_img)

Qair_mean = np.nanmean(imp_img[:,:,3])

print ("Qair : {:.8f}".format(Qair_mean))

Tair_mean = np.nanmean(imp_img[:,:,10])

print ("Qair : {:.8f}".format(Tair_mean))

print ("wind : {:.8f}".format(np.nanmean(imp_img[:,:,11])))
latitude=18.1429005246921; longitude=-65.4440010699994

overlay_image_on_puerto_rico(image,10,latitude,longitude, 9)

#overlay_image_on_puerto_rico(image,band_layer=7, lat=latitude,lon=longitude,zoom=8)
image = data_path + '/gldas/gldas_20180701_0600.tif'



plot_image_gldas(image, 'BuPu_r')
## Use this cell to define the collection to see



startdate = '2018-12-01'

finishdate = '2018-12-31'



#Sentinel-5P OFFL NO2: Offline Nitrogen Dioxide

ee_s5p_no2 = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')

              .select('NO2_column_number_density')

              .filterDate(startdate, finishdate)

             )

band_viz_s5p_no2 = {

    'min': 0,

    'max': 0.0002,

    'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}



#GFS: Global Forecast System 384-Hour Predicted Atmosphere Data

ee_gfs = (ee.ImageCollection('NOAA/GFS0P25')

          .select('v_component_of_wind_10m_above_ground')

          .filterDate(startdate, finishdate)

         )

band_viz_gfs = {

    'min': -40.0,

    'max': 35.0,

    'palette': ['blue', 'purple', 'cyan', 'green', 'yellow', 'red']}



#GLDAS-2.1: Global Land Data Assimilation System

ee_gldas = (ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H')

          .select('Wind_f_inst')

          .filterDate(startdate, finishdate)

         )

band_viz_gldas = {

    'min': 250.0,

    'max': 300.0,

    'palette': ['1303ff', '42fff6', 'f3ff40', 'ff5d0f']}
#Init Puerto Rico Map with power plants

lat=18.200178; lon=-66.664513 #puerto rico

Map = initMap(power_plants_df, lat, lon)



# Add the sentinel N2O layer to the map object.

Map.add_ee_layer(ee_s5p_no2.mean(), band_viz_s5p_no2, 'S5P_NO2')



# Add the GFS layer to the map object.

Map.add_ee_layer(ee_gfs.mean(), band_viz_gfs, 'GFS')



# Add the GLDAS layer to the map object.

Map.add_ee_layer(ee_gldas.mean(), band_viz_gldas, 'GLDAS')



# Add a layer control panel to the map.

Map.add_child(folium.LayerControl())



# Display the map.

Map