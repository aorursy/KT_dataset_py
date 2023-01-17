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
power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)
power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)
power_plants['latitude'] = power_plants['latitude'].astype(float)
a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8
power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 
lat=18.200178; lon=-66.664513
plot_points_on_map(power_plants,0,425,'latitude',lat,'longitude',lon,9)
power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()
power_plants_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh']]
image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180708T172237_20180714T190743.tif'
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
from kaggle_secrets import UserSecretsClient
from google.oauth2.credentials import Credentials
import ee
import folium

def add_ee_layer(self, ee_image_object, vis_params, name):
  # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
    tiles = map_id_dict['tile_fetcher'].url_format,
    attr = 'Map Data; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    # attr = 'Map Data; Google Earth Engine',
    name = name,
    overlay = True,
    control = True
  ).add_to(self)

def add_ee_layer_median(self, ee_image_object, vis_params,name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
    tiles = map_id_dict['tile_fetcher'].url_format,
    attr = 'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    name = name,
    overlay = True,
    control = True,
  ).add_to(self)

def plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom):
    # https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/ee-api-colab-setup.ipynb
    folium.Map.add_ee_layer = add_ee_layer
    vis_params = {
      'min': minimum_value,
      'max': maximum_value,
      'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red'],
      'opacity': 0.5
    }
    my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)
    s5p = ee.ImageCollection(dataset).filterDate(begin_date, end_date)
    my_map.add_ee_layer(s5p.mean().select(column), vis_params, 'Color')
    my_map.add_child(folium.LayerControl())
    display(my_map)
   
def plot_ee_data_on_map_mean(image,column,minimum_value,maximum_value,latitude,longitude,zoom):
    folium.Map.add_ee_layer = add_ee_layer_median
    vis_params = {
      'min': minimum_value,
      'max': maximum_value,
      'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red'],
      'opacity': 0.5
    }

    my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)
    my_map.add_ee_layer(image.select(column), vis_params, 'NO2')
    my_map.add_child(folium.LayerControl())
    display(my_map)
   
def plot_ee_Image_on_map(image,minimum_value,maximum_value,latitude,longitude,zoom):
    folium.Map.add_ee_layer = add_ee_layer_median
    vis_params = {
      'min': minimum_value,
      'max': maximum_value,
      'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red'],
      'opacity': 0.5
    }

    my_map = folium.Map(location=[latitude,longitude], zoom_start=zoom, height=500)
    # my_map.centerObject(aoi);
    my_map.add_ee_layer(image, vis_params, 'NO2')
    my_map.add_child(folium.LayerControl())
    display(my_map) 

def overlay_image_on_location(image,latitude,longitude,bounds):
    m = folium.Map([lat, lon], zoom_start=8)
    folium.raster_layers.ImageOverlay(
        image=image,
        bounds = bounds,
        colormap=lambda x: (1, 0, 0, x),
    ).add_to(m)
    return m

# Function to filter out images which do not fully cover the study area
def getCover(image):
    
    column='NO2_column_number_density'
    aoi= ee.Geometry.Polygon(
        [[[-56.23626061037187, -25.073217930024114],
          [-56.23626061037187, -28.77493701545809],
          [-48.08440514162187, -28.77493701545809],
          [-48.08440514162187, -25.073217930024114]]]);
    scale=7000
    
    # calculate the number of inputs 
    # reduced = image.reduceRegion()
    totPixels = ee.Number(ee.Image(image).reduceRegion(geometry=aoi,reducer=ee.Reducer.count(), 
                                                       crs=image.select(column).projection(),scale=scale).values().get(0))
    
    # Calculate the actual amount of pixels inside the aoi
    actPixels = ee.Number(ee.Image(image).reduceRegion(geometry=aoi,reducer=ee.Reducer.count(), 
                                                       scale=scale).values().get(0))
    
    # calculate the perc of cover
    percCover = ee.Number(actPixels).divide(totPixels).multiply(100).round();

    # number as output
    image = ee.Image(image).set('percCover', percCover);
    return image
ee.Authenticate()
!cat ~/.config/earthengine/credentials
!pip install sentinelsat
user_secret = "earth-engine" # Your user secret, defined in the add-on menu of the notebook editor
refresh_token = UserSecretsClient().get_secret(user_secret)
credentials = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=ee.oauth.TOKEN_URI,
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
        scopes=ee.oauth.SCOPES)
ee.Initialize(credentials=credentials)
dataset = "COPERNICUS/S5P/NRTI/L3_NO2"
column = 'NO2_column_number_density'
begin_date = '2019-08-01'
end_date = '2020-04-15'
minimum_value = 0.0
maximum_value = 0.0002
latitude = -23.44
longitude = -51.95
zoom = 5
plot_ee_data_on_map(dataset,column,begin_date,end_date,minimum_value,maximum_value,latitude,longitude,zoom)

from sentinelsat import  read_geojson, geojson_to_wkt

polygon= geojson_to_wkt(read_geojson('/kaggle/input/sc-geojson/geojs-42-mun.json', encoding='latin-1'))
print(polygon)
region = ee.Geometry.Polygon(
        [[[-56.23626061037187, -25.073217930024114],
          [-56.23626061037187, -28.77493701545809],
          [-48.08440514162187, -28.77493701545809],
          [-48.08440514162187, -25.073217930024114]]]);
L3_NO2 = ee.ImageCollection(dataset).filterDate('2020-04-10','2020-05-10').select('NO2_column_number_density')

aoi = region
coll_with_zero_flag = L3_NO2.map(algorithm=getCover)
coll_filt_clean = coll_with_zero_flag.filter(ee.Filter.gt('percCover', 70));
single_scene = coll_filt_clean.max();
mask = ee.Image.constant(255).clip(aoi)

plot_ee_Image_on_map(single_scene.updateMask(mask),minimum_value,maximum_value,latitude,longitude,zoom)
# Select Red and NIR bands, scale them, and sample 500 points.
# samp_fc = img.select(['B3','B4']).divide(10000).sample(scale=30, numPixels=500)
samp_fc = image.divide(10000).sample(region)

# Arrange the sample as a list of lists.
samp_dict = samp_fc.reduceColumns(ee.Reducer.toList(), ['NO2_column_number_density'])
print(samp_list)
samp_list = ee.List(samp_dict.get('list'))
print(samp_list)

# Save server-side ee.List as a client-side Python list.
samp_data = samp_list.getInfo()

# Display a scatter plot of Red-NIR sample pairs using matplotlib.
plt.scatter(samp_data[0], samp_data[1], alpha=0.2)
plt.xlabel('Red', fontsize=12)
plt.ylabel('NIR', fontsize=12)
plt.show()

plot_ee_data_on_map_mean(image,column,minimum_value,maximum_value,latitude,longitude,zoom)
# r = 1
# region = [longitude-r,latitude-r,longitude+r,latitude+r]
# region=ee.Geometry.Rectangle(region);
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