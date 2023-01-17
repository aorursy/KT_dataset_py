import numpy as np # linear algebra

import folium



# Connect to Earth Engine

import ee

from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials



# Trigger the authentication flow.

#ee.Authenticate()



# Retrieve your refresh token.

#!cat ~/.config/earthengine/credentials



user_secret = "AJR_EIE_test" # Your user secret, defined in the add-on menu of the notebook editor

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



def add_ee_layer(self, ee_image_object, vis_params, name):

    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)

    folium.raster_layers.TileLayer(

        tiles = map_id_dict['tile_fetcher'].url_format,

        attr = "Map Data © Google Earth Engine",

        name = name,

        overlay = True,

        control = True

    ).add_to(self)

    

# modified from https://www.kaggle.com/paultimothymooney/how-to-get-started-with-the-earth-engine-data

# zoom_country=True when showing Puerto Rico

# takes default lat and long (unless they are overwritten) and zoom=8

def plot_ee_data_on_map(add_ee_layer, begin_date, end_date,

                        min_value, max_value, opacity=1.0, lat=18.233, long=-66.279, 

                        zoom_country=True, res=0.01):

    

    product = "NO2"

    dataset = "COPERNICUS/S5P/OFFL/L3_NO2"

    column = 'tropospheric_NO2_column_number_density'

    

    if zoom_country: # zoom at the country level, default lat and long

        zoom_start = 8

        lat1 = lat-0.33; long1 = long-1.06

        lat2 = lat+0.33; long2 = long+1.06

    else: # zoom at a Power Plant level

        zoom_start = 13

        lat1 = lat-res/2; long1 = long-res/2

        lat2 = lat+res/2; long2 = long+res/2

    rectangle = ee.Geometry.Rectangle([long1, lat1, long2, lat2]) # (x, y) math style   

        

    Map = folium.Map(location=[lat, long], zoom_start=zoom_start) # (y, x) geo style

    folium.Map.add_ee_layer = add_ee_layer



    sat_image = (ee.ImageCollection(dataset)

           .select(column)

           .filterDate(begin_date, end_date)

           .mean()

          )

    

    vis_params = {

      'min': min_value,

      'max': max_value,

      'opacity': opacity,

      'palette': ['green', 'blue', 'yellow', 'red']}

        

    Map.add_ee_layer(sat_image.clip(rectangle), vis_params, product)

    Map.add_child(folium.LayerControl())

    display(Map)

    return sat_image    
# A "pixel" around San Juan Power Plant

res = 0.01 # pixel resolution in arc degrees

long = -66.1045; lat = 18.427 # San Juan



begin_date = '2019-05-03'; end_date = '2019-05-04'

min_value = 0.00001; max_value = 0.000025



sat_image = plot_ee_data_on_map(add_ee_layer, begin_date, end_date, min_value, max_value, 

                        zoom_country=False, long=long, lat=lat, res=res)
res = 0.01

lat1 = lat-res/2; long1 = long-res/2

lat2 = lat+res/2; long2 = long+res/2



from haversine import haversine # distance in km (default) between points in UTM coordinates

p1 = (lat1, long1) # (y, x) geo style

p2 = (lat1, long2)

p3 = (lat2, long1)

p4 = (lat2, long2)

print("Horizontal pixel side", haversine(p1, p2), "km")

print("Vertical pixel side", haversine(p1, p3), "km")
import rasterio as rio

import os



s5p_file = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20190501T161114_20190507T174400.tif'

def preview_meta_data(file_name):

    with rio.open(file_name) as img_filename:

        print('Metadata for: ',file_name)

        print('Bounding Box:',img_filename.bounds)

        print('Resolution:',img_filename.res)

        print('Tags:',img_filename.tags())

        print('More Tags:',img_filename.tags(ns='IMAGE_STRUCTURE'))

        print('Number of Channels =',img_filename.count,'\n')



preview_meta_data(s5p_file)
rectangle = ee.Geometry.Rectangle([long1, lat1, long2, lat2]) # the pixel



collection = (ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')

  .filterDate(begin_date, end_date))



count = collection.size()

print('Count: ', str(count.getInfo())+'\n')
image = collection.first()

band_arrs = image.sampleRectangle(rectangle);

# Get individual band arrays.

band_arr = band_arrs.get('tropospheric_NO2_column_number_density')

np_arr = np.array(band_arr.getInfo()) 
orbitStats = collection.aggregate_stats("ORBIT")

minOrbit = orbitStats.getInfo()['values']['min']

maxOrbit = orbitStats.getInfo()['values']['max']

arrayList = []

for orbit in range(minOrbit, maxOrbit+1):

    index = orbit - minOrbit

    filtered = collection.filterMetadata('ORBIT', 'equals', orbit);

    image = filtered.first()

    try:

        date = image.date()

        # only arrives here in case of no error: incomplete image throws exception

        band_arrs = image.sampleRectangle(rectangle);

        # Get individual band arrays.

        band_arr = band_arrs.get('tropospheric_NO2_column_number_density')



        # Transfer the arrays from server to client and cast as np array.

        try:

            np_arr = np.array(band_arr.getInfo()) 

            # only arrives here in case of no error

            arrayList.append(np_arr)

            print("Catch!", date.format().getInfo())

        except ee.EEException:

            print("Bad luck", date.format().getInfo())

    except ee.EEException:

            # incomplete image

            print("Very bad luck, incomplete image", orbit)

        

print("Fin")
print(arrayList[0].shape, arrayList[0])