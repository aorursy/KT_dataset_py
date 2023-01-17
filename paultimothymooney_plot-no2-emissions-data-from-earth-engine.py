import numpy as np

import pandas as pd

import folium

from folium import plugins



def initMap(lat, lon):

    location = [lat, lon]

    Map = folium.Map(location=location, zoom_start=7)

    return Map



def add_ee_layer(self, ee_image_object, vis_params, name):

  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)

  folium.raster_layers.TileLayer(

    tiles = map_id_dict['tile_fetcher'].url_format,

    attr = "Map Data © Google Earth Engine",

    name = name,

    overlay = True,

    control = True

  ).add_to(self)

    

folium.Map.add_ee_layer = add_ee_layer



# Connect to Earth Engine

import ee

from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials



# Trigger the authentication flow.

#ee.Authenticate()



# Retrieve your refresh token.

#!cat ~/.config/earthengine/credentials



user_secret = "earth_engine" # Your user secret, defined in the add-on menu of the notebook editor

refresh_token = UserSecretsClient().get_secret(user_secret)

credentials = Credentials(

        None,

        refresh_token=refresh_token,

        token_uri=ee.oauth.TOKEN_URI,

        client_id=ee.oauth.CLIENT_ID,

        client_secret=ee.oauth.CLIENT_SECRET,

        scopes=ee.oauth.SCOPES)



# Initialize GEE

ee.Initialize(credentials=credentials)
startdate = '2019-10-01'

finishdate = '2019-10-14'

latitude = 18.20

longitude = -66.66
#Sentinel-5P OFFL NO2: Offline Nitrogen Dioxide

ee_s5p_no2 = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')

              .select('NO2_column_number_density')

              .filterDate(startdate, finishdate)

             )

band_viz_s5p_no2 = {

    'min': 0,

    'max': 0.00015,

    'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']}

Map = initMap(lat = latitude, lon = longitude)

Map.add_ee_layer(ee_s5p_no2.mean(), band_viz_s5p_no2, 'S5P_NO2')

Map.add_child(folium.LayerControl())

Map
Map = initMap(lat = latitude, lon = longitude)

Map