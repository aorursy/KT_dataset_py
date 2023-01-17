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



import warnings

warnings.filterwarnings('ignore')



#Code borrowed from Gilles Poulain 

global_power_plants = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')

global_power_plants.head(3)
relevant_cols = ['name','capacity_mw','commissioning_year','estimated_generation_gwh','owner','primary_fuel','wepp_id','.geo','gppd_idnr']

pr_plants = global_power_plants[relevant_cols]

pr_plants.head()
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



pr_plants['lon'] = pr_plants['.geo'].map(get_lon_from_geo)

pr_plants['lat'] = pr_plants['.geo'].map(get_lat_from_geo)



pr_plants.drop(columns=['.geo'], inplace=True)



geometry_pr_plants = [Point(x,y) for x,y in zip(pr_plants['lon'], pr_plants['lat'])]

pr_plants_gdf = gpd.GeoDataFrame(pr_plants, crs = {'init': 'epsg: 4326'}, geometry = geometry_pr_plants)



# from https://www.kaggle.com/maxlenormand/simplified-emission-for-each-plant-wip

buffered_pr_plants = pr_plants.copy()

buffered_pr_plants['geometry'] = pr_plants_gdf.geometry.buffer(0.05)



pr_plants.head(3)
df_fossil = pr_plants[(pr_plants.primary_fuel == 'Oil')|(pr_plants.primary_fuel == 'Gas')]

df_renew = pr_plants[(pr_plants.primary_fuel == 'Hydro')|(pr_plants.primary_fuel == 'Solar')|(pr_plants.primary_fuel == 'Wind')]
df_fossil
df_renew
sentinel_path = "../input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/"

s5p_files = os.listdir(sentinel_path)

s5p_dates = [datetime.strptime(g[:16], 's5p_no2_%Y%m%d') for g in s5p_files]



ctr = 0 

for city_name in df_fossil.name:

    ctr += 1

    df_plant = pr_plants_gdf[pr_plants_gdf['name'] == city_name]



    dates = []

    stats = []

    for s5p_file in s5p_files:

        image_path = sentinel_path + s5p_file

        date = datetime.strptime(s5p_file[:16], 's5p_no2_%Y%m%d')

        stat = rasterstats.zonal_stats(df_plant.geometry.to_json(),

                                       image_path,

                                       band=2, #2: tropospheric_NO2_column_number_density

                                       stats=['mean'])

        stat = stat[0] # get location of pp

        stat = stat['mean'] # retrieve stat

        dates.append(date)

        stats.append(stat)

    if ctr == 1:

        results = pd.DataFrame(index=dates, data=stats, columns=[city_name])

    else: 

        results[city_name] = stats

results.plot()
import matplotlib

import matplotlib.pyplot as plt



fossil_results = results



fig = plt.figure()

fig = matplotlib.pyplot.gcf()

fig.set_size_inches(7,13)

plt.rcParams["axes.labelweight"] = "bold"

plt.rcParams["font.size"] = 14

plt.rcParams["font.weight"] = "bold"

plt.rcParams["figure.figsize"] = (20,8)

fossil_results.plot()
ctr = 0 

for city_name in df_renew.name:

    ctr += 1

    df_plant = pr_plants_gdf[pr_plants_gdf['name'] == city_name]



    dates = []

    stats = []

    for s5p_file in s5p_files:

        image_path = sentinel_path + s5p_file

        date = datetime.strptime(s5p_file[:16], 's5p_no2_%Y%m%d')

        stat = rasterstats.zonal_stats(df_plant.geometry.to_json(),

                                       image_path,

                                       band=2, #2: tropospheric_NO2_column_number_density

                                       stats=['mean'])

        stat = stat[0] # get location of pp

        stat = stat['mean'] # retrieve stat

        dates.append(date)

        stats.append(stat)

    if ctr == 1:

        results = pd.DataFrame(index=dates, data=stats, columns=[city_name])

    else: 

        results[city_name] = stats
renewable_results = results

renewable_results = renewable_results.mask(renewable_results > 0.0001, 0.0001)

renewable_results.plot()
(fossil_results.mean()*1000000).hist()
(renewable_results.mean()*1000000).hist()
#Despite fossil plants having slightly higher overall mean, 

#there is no significant trend between emissions and weather season or plant type (fossil & renewable). 

#Other factors such as population size might

#play a far greater role, making it difficult to isolate plant's emission impact 