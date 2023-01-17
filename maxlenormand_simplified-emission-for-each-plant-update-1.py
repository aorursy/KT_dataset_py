!pip install rasterstats --quiet
import numpy as np

import pandas as pd

import os

from tqdm import tqdm



# Plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns
# Geospatial libraries that we will be using for this

import rasterio

import rasterstats

import folium

import geopandas as gpd

from shapely.geometry import Point
power_plants_path = '/kaggle/input/geolocated-power-plants-geojson/Geolocated_gppd_120_pr.geojson'

power_plants = gpd.read_file(power_plants_path)
buffered_power_plants = power_plants

buffered_power_plants['geometry'] = power_plants.geometry.buffer(0.1)
# Plot on the map

lat=18.200178; lon=-66.664513



def plot_polygons_on_Map(geodataframe,

                       original_latitude = lat,

                       original_longitude = lon,

                       zoom=9):

    plot = folium.Map(location = (original_latitude, original_longitude), zoom_start=zoom)





    geojson = folium.GeoJson(geodataframe,

                            style_function=lambda x: {'Color':'red',

                                                      'fillColor':'yellow' if x['properties']['primary_fuel']=='Solar'  else 'red'})

    popup = folium.Popup(str(geodataframe.primary_fuel))

    popup.add_to(geojson)

    geojson.add_to(plot)



    return(plot)
plot_polygons_on_Map(buffered_power_plants)
# Make a dataframe containing all the S5 raster paths and dates



from datetime import datetime



df_s5 = pd.DataFrame()



files=[]

captured_datetime = []

for dirname, _, filenames in os.walk('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))

        captured_datetime.append(datetime.strptime(filename.split('_')[-2],'%Y%m%dT%H%M%S').date())

        

df_s5['S5_path'] = files

df_s5['Captured_datetime'] = captured_datetime



# Sort dataframe by ascending time

df_s5.sort_values('Captured_datetime', inplace=True)

df_s5.reset_index(inplace=True, drop=True)



df_s5.head()
print(f"There are {df_s5.shape[0]} Sentinel 5 images from {df_s5['Captured_datetime'][0]} to {df_s5['Captured_datetime'][df_s5.shape[0]-1]}")
test_S5_raster_path = df_s5['S5_path'][0]
def return_bands(file_name):

    # Function from the organizers: https://www.kaggle.com/paultimothymooney/explore-image-metadata-s5p-gfs-gldas

    image = rasterio.open(file_name)

    for i in image.indexes:

        desc = image.descriptions[i-1]

        print(f'{i}: {desc}')

        

return_bands(test_S5_raster_path)
power_plants_stats = rasterstats.zonal_stats(power_plants_path,

                                                 test_S5_raster_path,

                                                 band=1,

                                                 stats=['mean'])
# Plotting the first few elements

power_plants_stats[:5]
cols_to_keep = ['system:index', 'capacity_mw', 'commissioning_year', 'estimated_generation_gwh', 'primary_fuel',  'geometry']

simplified_emissions_factor_df = buffered_power_plants.copy(deep=True)

simplified_emissions_factor_df = simplified_emissions_factor_df[cols_to_keep]



simplified_emissions_factor_df['mean_N02'] = [x['mean'] for x in power_plants_stats]
simplified_emissions_factor_df['Simpl_Emiss_Factor'] = simplified_emissions_factor_df['mean_N02'] / simplified_emissions_factor_df['estimated_generation_gwh']
MULTIPLICATION_FACTOR = 100000



plt.figure()

sns.distplot(simplified_emissions_factor_df['Simpl_Emiss_Factor']*MULTIPLICATION_FACTOR,

             kde=False, 

             label=f'SEF x {MULTIPLICATION_FACTOR}')

plt.legend()

plt.title('Simplified Emission Factor for each plant (N02 / Estimated Emsission)')
SUBSET_TO_TEST = df_s5.shape[0]



N02_measurments_df = pd.DataFrame({'index_power_plant':power_plants['system:index']})



for s5_image in tqdm(range(SUBSET_TO_TEST)):

    S5_raster_stats = rasterstats.zonal_stats(power_plants_path,

                                                     df_s5['S5_path'][s5_image],

                                                     band=1,

                                                     stats=['mean'])

    list_of_S5_raster_stats = [plant['mean'] for plant in S5_raster_stats]

    

    N02_measurments_df[df_s5['Captured_datetime'][s5_image]] = list_of_S5_raster_stats
N02_measurments_df.head()
print(f'There are {N02_measurments_df.iloc[0].isna().sum()} NaNs ({np.round((N02_measurments_df.iloc[0].isna().sum() / N02_measurments_df.shape[1]) * 100, 2)}% of data)')
# Removing NaNs



test_power_plant_time_series = N02_measurments_df.iloc[0].dropna()

test_power_plant_time_series.isna().sum()
plt.figure(figsize=(15, 10))

plt.plot(list(test_power_plant_time_series[1:]))

plt.title(f'N02 measurement of plant {N02_measurments_df["index_power_plant"][0]}')
# Making a dataframe only keeping the info related to the enegery production



power_plant_type_infos = power_plants[['capacity_mw', 'commissioning_year', 'estimated_generation_gwh', 'primary_fuel']]

power_plant_type_infos.head()
np.unique(power_plant_type_infos.primary_fuel, return_counts=True)
id_coal = power_plant_type_infos.loc[power_plant_type_infos['primary_fuel']=='Coal'].index.values[0]

id_coal
# Looking for coal and solar



# --- Coal ----



# Removing NaNs

id_coal = power_plant_type_infos.loc[power_plant_type_infos['primary_fuel']=='Coal'].index.values[0]

coal_plants_stats = N02_measurments_df.iloc[id_coal].dropna()

print(f'There are {coal_plants_stats.isna().sum()} NaNs left')



# --- Solar ----



# Removing NaNs (only keeping 1 of the solar plants)

id_solar = power_plant_type_infos.loc[power_plant_type_infos['primary_fuel']=='Solar'].index.values[0]

solar_plants_stats = N02_measurments_df.iloc[id_solar].dropna()

print(f'There are {solar_plants_stats.isna().sum()} NaNs left')





plt.figure(figsize=(30, 10))



plt.plot(list(coal_plants_stats[1:]), label='Coal', color='brown')

plt.plot(list(solar_plants_stats[1:]), label='Solar', color='blue')



plt.legend()

plt.title(f'N02 measurement of the only Coal and 1 solar plant')

plt.show()





plants_of_interest = buffered_power_plants.iloc[[id_solar, id_coal]]

plot_polygons_on_Map(geodataframe=plants_of_interest)