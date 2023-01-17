!pip install rasterstats --quiet
## Importing necessary libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)



from ast import literal_eval

from datetime import datetime, timedelta

import os, random



# Plotting geographical data

import folium

from folium import plugins

import rasterio as rio

import rasterstats

import geopandas as gpd

from shapely.geometry import Point
global_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

global_power_plants.head(3)
global_power_plants.info()
global_power_plants.describe().T
# Columns with only 0 or NaN values

to_drop = ["generation_gwh_2013", 

           "generation_gwh_2014", 

           "generation_gwh_2015", 

           "generation_gwh_2016",

           "generation_gwh_2017", 

           "other_fuel1",

           "other_fuel2",

           "other_fuel3",

           "year_of_capacity_data"]



global_power_plants = global_power_plants.drop(to_drop, axis=1)
global_power_plants['country'].unique()
global_power_plants['country_long'].unique()
global_power_plants['geolocation_source'].unique()
# Columns with all same values

to_drop = ["country", 

           "country_long", 

           "geolocation_source"

          ]



global_power_plants = global_power_plants.drop(to_drop, axis=1)
def get_lon_from_geo(str_):

    dict_ = literal_eval(str_)

    coordinates = dict_['coordinates']

    lon = coordinates[0]

    return lon



def get_lat_from_geo(str_):

    dict_ = literal_eval(str_)

    coordinates = dict_['coordinates']

    lat = coordinates[1]

    return lat



global_power_plants['lon'] = global_power_plants['.geo'].map(get_lon_from_geo)

global_power_plants['lat'] = global_power_plants['.geo'].map(get_lat_from_geo)



global_power_plants.drop(columns=['.geo'], inplace=True)



global_power_plants.head(3)
geometry_power_plants = [Point(x,y) for x,y in zip(global_power_plants['lon'], global_power_plants['lat'])]

global_power_plants_gdf = gpd.GeoDataFrame(global_power_plants, crs = {'init': 'epsg: 4326'}, geometry = geometry_power_plants)



global_power_plants_gdf.plot()
fig, ax = plt.subplots(1,2,figsize=(12,5))



sns.countplot(x="primary_fuel",

              data=global_power_plants,

              ax=ax[0]

             )

sns.stripplot(x="primary_fuel",

            y="capacity_mw",

            data=global_power_plants,

            ax=ax[1]

           )



plt.tight_layout()
sns.barplot(x='primary_fuel',

            y="capacity_mw",

            data=global_power_plants.groupby(['primary_fuel']).sum().reset_index())
sns.barplot(x="primary_fuel",

            y="estimated_generation_gwh",

            data=global_power_plants.groupby(['primary_fuel']).sum().reset_index())
sns.barplot(x="primary_fuel",

            y="estimated_generation_gwh",

            data=global_power_plants[global_power_plants['primary_fuel'] != 'Coal'].groupby(['primary_fuel']).sum().reset_index())
solar_wind_power_plants = global_power_plants[(global_power_plants['primary_fuel'] == 'Solar') |

                                              (global_power_plants['primary_fuel'] == 'Wind')]



solar_wind_power_plants
def initMap(df, lat, lon):

    location = [lat, lon]

    Map = folium.Map(location=location, zoom_start=9)

    

    fuelColor = {'Coal': 'darkred',

                 'Oil': 'black',

                 'Gas': 'lightgray',

                 'Hydro': 'lightblue',

                 'Solar': 'orange',

                 'Wind': 'green'

                }

    

    for _, row in df.iterrows():

        fuel = row['primary_fuel']

        capacity = row['capacity_mw']

        generation = row['estimated_generation_gwh']

        locationpp = [row['lat'], row['lon']]

        popup = "{} {}MW capacity, {}GWh generated".format(fuel,capacity,generation)

        color = fuelColor[fuel]

        folium.Marker(locationpp,

                      popup=popup,

                      icon=folium.Icon(color=color, icon_color='white', icon='bolt', prefix='fa')

                     ).add_to(Map)  



    return Map
lat=18.200178; lon=-66.664513 #puerto rico

Map = initMap(global_power_plants, lat, lon)   

Map
sentinel_path = "../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/"

examples = [random.choice(os.listdir(sentinel_path)) for _ in range(4)]

examples
image_name = random.choice(os.listdir(sentinel_path))

image_path = sentinel_path + image_name

image = rio.open(image_path)



bands = image.count

print(f"There are {bands} bands")



for i in image.indexes:

    desc = image.descriptions[i-1]

    print(f'{i}: {desc}')



print('\n')

print(f'Example of:{image_name}')



fig, ax = plt.subplots(3,4,figsize=(20,6))

band=1

for n in range(3):

    for m in range(4):

        image_bandi = image.read(band)

        desc = image.descriptions[band-1]

        ax[n,m].set_title(desc)

        ax[n,m].imshow(image_bandi, cmap="Reds")

        ax[n,m].grid(False)

        band += 1

        

fig.tight_layout()
gfs_path = "../input/ds4g-environmental-insights-explorer/eie_data/gfs/"

examples = [random.choice(os.listdir(gfs_path)) for _ in range(4)]

examples
image_name = random.choice(os.listdir(gfs_path))

image_path = gfs_path + image_name

image = rio.open(image_path)



bands = image.count

print(f"There are {bands} bands")



for i in image.indexes:

    desc = image.descriptions[i-1]

    print(f'{i}: {desc}')



print('\n')

print(f'Example of:{image_name}')



fig, ax = plt.subplots(3,2,figsize=(15,6))

band=1

for n in range(3):

    for m in range(2):

        image_bandi = image.read(band)

        desc = image.descriptions[band-1]

        ax[n,m].set_title(desc)

        ax[n,m].imshow(image_bandi, cmap="Reds")

        ax[n,m].grid(False)

        band += 1

        

fig.tight_layout()
gldas_path = "../input/ds4g-environmental-insights-explorer/eie_data/gldas/"

examples = [random.choice(os.listdir(gldas_path)) for _ in range(4)]

examples
image_name = random.choice(os.listdir(gldas_path))

image_path = gldas_path + image_name

image = rio.open(image_path)



bands = image.count

print(f"There are {bands} bands")



for i in image.indexes:

    desc = image.descriptions[i-1]

    print(f'{i}: {desc}')



print('\n')

print(f'Example of:{image_name}')



fig, ax = plt.subplots(3,4,figsize=(20,6))

band=1

for n in range(3):

    for m in range(4):

        image_bandi = image.read(band)

        desc = image.descriptions[band-1]

        ax[n,m].set_title(desc)

        ax[n,m].imshow(image_bandi, cmap="Reds")

        ax[n,m].grid(False)

        band += 1

        

fig.tight_layout()
gldas_files = os.listdir(gldas_path)

gldas_dates = [datetime.strptime(g, 'gldas_%Y%m%d_%H%M.tif') for g in gldas_files]



gfs_files = os.listdir(gfs_path)

gfs_dates = [datetime.strptime(g, 'gfs_%Y%m%d%H.tif') for g in gfs_files]



s5p_files = os.listdir(sentinel_path)

s5p_dates = [datetime.strptime(g[:16], 's5p_no2_%Y%m%d') for g in s5p_files]





all_dates = (pd.DataFrame(columns=['dataset', 'datetime'])

             .append(pd.DataFrame(gldas_dates, columns=['datetime'])

                     .assign(dataset = 'gldas'), sort=True)

             .append(pd.DataFrame(gfs_dates, columns=['datetime'])

                     .assign(dataset = 'gfs'), sort=True)

             .append(pd.DataFrame(s5p_dates, columns=['datetime'])

                     .assign(dataset = 's5p'), sort=True)

            ).assign(date = lambda x: x.datetime.apply(lambda x: x.date()))
all_dates.groupby('dataset').date.agg(

    min=min,

    max=max,

    measurement_period=  lambda x: (x.max()-x.min()).days+1,

    measurement_count= 'count',

    measurements_per_day= lambda x: x.count() / ((x.max()-x.min()).days+1)

).T
daily_data = (pd.date_range('2018-07-01', '2019-06-30').to_frame() # Get the date index to work with

              .merge( #Merge with all_dates

                  all_dates.groupby(['dataset', 'date']).date.count().unstack(level=0),

                  left_index=True,

                  right_index=True)

              .drop(columns=[0], axis=1) #remove col used for merge

             )



sns.heatmap(daily_data.transpose())

fig = plt.gcf()

fig.set_size_inches(11,3)
# from https://www.kaggle.com/maxlenormand/simplified-emission-for-each-plant-wip

buffered_power_plants = global_power_plants_gdf.copy()

buffered_power_plants['geometry'] = global_power_plants_gdf.geometry.buffer(0.05)
lat=18.200178; lon=-66.664513 #puerto rico

Map = initMap(global_power_plants, lat, lon)   



for power_plants in range(buffered_power_plants.shape[0]):

    folium.GeoJson(buffered_power_plants.geometry[power_plants]).add_to(Map)



Map
oil_pp_Cambalache = global_power_plants_gdf[global_power_plants_gdf['name'] == 'Cambalache']

oil_pp_Cambalache
dates = []

stats = []

for s5p_file in s5p_files:

    image_path = sentinel_path + s5p_file

    date = datetime.strptime(s5p_file[:16], 's5p_no2_%Y%m%d')

    stat = rasterstats.zonal_stats(oil_pp_Cambalache.geometry.to_json(),

                                   image_path,

                                   band=2, #2: tropospheric_NO2_column_number_density

                                   stats=['mean'])

    stat = stat[0] # get location of pp

    stat = stat['mean'] # retrieve stat

    dates.append(date)

    stats.append(stat)



results = pd.DataFrame(index=dates, data=stats, columns=['oil'])

results.plot()

plt.title('2: tropospheric_NO2_column_number_density over time in gas_pp_Cambalache')
solar_pp_Oriana = global_power_plants_gdf[global_power_plants_gdf['name'] == 'Oriana Solar Farm']

solar_pp_Oriana
dates = []

stats = []

for s5p_file in s5p_files:

    image_path = sentinel_path + s5p_file

    date = datetime.strptime(s5p_file[:16], 's5p_no2_%Y%m%d')

    stat = rasterstats.zonal_stats(solar_pp_Oriana.geometry.to_json(),

                                   image_path,

                                   band=2, #2: tropospheric_NO2_column_number_density

                                   stats=['mean'])

    stat = stat[0] # get location of pp

    stat = stat['mean'] # retrieve stat

    dates.append(date)

    stats.append(stat)



results_solar = pd.DataFrame(index=dates, data=stats, columns=['solar'])

results_solar.plot()

plt.title('2: tropospheric_NO2_column_number_density over time in gas_pp_Oriana')
results['solar'] = results_solar['solar']

results['difference'] = results['oil'] - results['solar']

results.plot()
mean_diff_oil_solar = results['difference'].mean()

print(f'Mean difference in tropospheric_NO2_column_number_density over a year: {mean_diff_oil_solar}')
fossil_power_plants_gdf = global_power_plants_gdf[(global_power_plants_gdf['primary_fuel'] == 'Oil') |

                                                  (global_power_plants_gdf['primary_fuel'] == 'Gas') |

                                                  (global_power_plants_gdf['primary_fuel'] == 'Coal')]



stats_all = []

for idx, fossil_power_plant in fossil_power_plants_gdf.iterrows():

    dates = []

    stats = []

    for s5p_file in s5p_files:

        image_path = sentinel_path + s5p_file

        date = datetime.strptime(s5p_file[:16], 's5p_no2_%Y%m%d')

        stat = rasterstats.zonal_stats(fossil_power_plant.geometry,

                                       image_path,

                                       band=2, #2: tropospheric_NO2_column_number_density

                                       stats=['mean'])

        stat = stat[0] # get location of pp

        stat = stat['mean'] # retrieve stat

        dates.append(date)

        stats.append(stat)

    name = fossil_power_plant['gppd_idnr']

    results_fossil_tmp = pd.DataFrame(index=dates, data=stats, columns=[name])

    stats_all.append(results_fossil_tmp)



results_fossil = pd.concat(stats_all, axis=1)

results_fossil.plot()
all_diff = []

for idnr in results_fossil.columns:

    diff = results_fossil[idnr] - results_solar['solar']

    #Add more infos from fossil_power_plants_gdf

    fossil_power_plant = fossil_power_plants_gdf[fossil_power_plants_gdf['gppd_idnr'] == idnr]

    capacity = float(fossil_power_plant['capacity_mw'])

    generatedpw = float(fossil_power_plant['estimated_generation_gwh'])

    fuel = fossil_power_plant['primary_fuel'].item()

    commissioning_year = int(fossil_power_plant['commissioning_year'])

    lat = float(fossil_power_plant['lat'])

    lon = float(fossil_power_plant['lon'])

    #put infos in a list

    all_diff.append([idnr, commissioning_year, fuel, capacity, generatedpw, diff.mean(), lat, lon])



all_diff_df = pd.DataFrame(all_diff, columns=['gppd_idnr', 'commissioning_year', 'primary_fuel', 'capacity_mw', 'estimated_generation_gwh', 'EF_N2O', 'lat', 'lon'])



geometry_power_plants = [Point(x,y) for x,y in zip(all_diff_df['lon'], all_diff_df['lat'])]

fossil_power_plants_EF_gdf = gpd.GeoDataFrame(all_diff_df, crs = {'init': 'epsg: 4326'}, geometry = geometry_power_plants)

fossil_power_plants_EF_gdf.head(3)
sns.scatterplot(x="estimated_generation_gwh", y="EF_N2O",

                hue="primary_fuel", size="capacity_mw",

                data=all_diff_df[all_diff_df['primary_fuel'] != 'Coal'])

# control x and y limits

plt.ylim(-0.000006, 0.000007)

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fossil_power_plants_EF_gdf.plot(column='EF_N2O', legend=True)
import ee

from kaggle_secrets import UserSecretsClient

from google.oauth2.credentials import Credentials



# Trigger the authentication flow.

#ee.Authenticate()
#!cat ~/.config/earthengine/credentials
user_secret = "gee" # Your user secret, defined in the add-on menu of the notebook editor

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
# Define a method for displaying Earth Engine image tiles to folium map.

def add_ee_layer(self, ee_image_object, vis_params, name):

  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)

  folium.raster_layers.TileLayer(

    tiles = map_id_dict['tile_fetcher'].url_format,

    attr = "Map Data © Google Earth Engine",

    name = name,

    overlay = True,

    control = True

  ).add_to(self)

    

# Add EE drawing method to folium.

folium.Map.add_ee_layer = add_ee_layer
## Use this cell to define the collection to see



startdate = '2018-07-01'

finishdate = '2018-07-03'



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

          .select('temperature_2m_above_ground')

          .filterDate(startdate, finishdate)

         )

band_viz_gfs = {

    'min': -40.0,

    'max': 35.0,

    'palette': ['blue', 'purple', 'cyan', 'green', 'yellow', 'red']}



#GLDAS-2.1: Global Land Data Assimilation System

ee_gldas = (ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H')

          .select('AvgSurfT_inst')

          .filterDate(startdate, finishdate)

         )

band_viz_gldas = {

    'min': 250.0,

    'max': 300.0,

    'palette': ['1303ff', '42fff6', 'f3ff40', 'ff5d0f']}
## Use this cell to add GEE layer to folium



#Init Puerto Rico Map with power plants

lat=18.200178; lon=-66.664513 #puerto rico

Map = initMap(global_power_plants, lat, lon)



# Add the sentinel N2O layer to the map object.

Map.add_ee_layer(ee_s5p_no2.mean(), band_viz_s5p_no2, 'S5P_NO2')



# Add the GFS layer to the map object.

#Map.add_ee_layer(ee_gfs.mean(), band_viz_gfs, 'GFS')



# Add the GLDAS layer to the map object.

Map.add_ee_layer(ee_gldas.mean(), band_viz_gldas, 'GLDAS')



# Add a layer control panel to the map.

Map.add_child(folium.LayerControl())



# Display the map.

Map