import pandas as pd

import folium
#Load the Data

df_apr14 = pd.read_csv('../input/uber-raw-data-apr14.csv')

df_may14 = pd.read_csv('../input/uber-raw-data-may14.csv')

df_jun14 = pd.read_csv('../input/uber-raw-data-jun14.csv')

df_jul14 = pd.read_csv('../input/uber-raw-data-jul14.csv')

df_aug14 = pd.read_csv('../input/uber-raw-data-aug14.csv')

df_sep14 = pd.read_csv('../input/uber-raw-data-sep14.csv')

df_janjune15 = pd.read_csv('../input/uber-raw-data-janjune-15.csv')
#Create Basemap that center in NYC

def generateBaseMap(default_location=[40.693943, -73.985880], default_zoom_start=12):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map

base_map = generateBaseMap()

base_map
df_14 = pd.concat([df_apr14, df_may14, df_jun14, df_jul14, df_aug14, df_sep14], sort=False, ignore_index=True)

df_14.head()
df_14.info()
df_14['Date/Time'] = pd.to_datetime(df_14['Date/Time'], format='%m/%d/%Y %H:%M:%S')
df_14['month'] = df_14['Date/Time'].apply(lambda x: x.month)

df_14['week'] = df_14['Date/Time'].apply(lambda x: x.week)

df_14['day'] = df_14['Date/Time'].apply(lambda x: x.day)

df_14['hour'] = df_14['Date/Time'].apply(lambda x: x.hour)
df_14.head()
#We first want to create a heatmap Uber Pickups in April from 12am to 5am.





from folium import plugins

from folium.plugins import HeatMap



df_copy = df_14[(df_14.hour < 5) & (df_14.month == 4)].copy()

df_copy['count'] = 1
df_hour_list = []

for hour in df_copy.hour.sort_values().unique():

    df_hour_list.append(df_copy.loc[df_copy.hour == hour, ['Lat', 'Lon', 'count']].groupby(['Lat', 'Lon']).sum().reset_index().values.tolist())

from folium.plugins import HeatMapWithTime

base_map = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_hour_list, radius=5, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map)

base_map
#create a heatmap Uber Pickups in April from 5am to 9am.
df_copy2 = df_14[(df_14.hour >= 5) & (df_14.hour < 9) & (df_14.month == 4)].copy()

df_copy2['count'] = 1
df_hour_list2 = []

for hour in df_copy2.hour.sort_values().unique():

    df_hour_list2.append(df_copy2.loc[df_copy2.hour == hour, ['Lat', 'Lon', 'count']].groupby(['Lat', 'Lon']).sum().reset_index().values.tolist())

base_map_59 = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_hour_list2, radius=5, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, 

                min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map_59)

base_map_59
#create a heatmap Uber Pickups in April from 9am to 11am.



df_copy3 = df_14[(df_14.hour >= 9) & (df_14.hour < 11) & (df_14.month == 4)].copy()

df_copy3['count'] = 1
df_hour_list3 = []

for hour in df_copy3.hour.sort_values().unique():

    df_hour_list3.append(df_copy3.loc[df_copy3.hour == hour, ['Lat', 'Lon', 'count']].groupby(['Lat', 'Lon']).sum().reset_index().values.tolist())

base_map_911 = generateBaseMap(default_zoom_start=11)

HeatMapWithTime(df_hour_list3, radius=5, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, 

                min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map_911)

base_map_911