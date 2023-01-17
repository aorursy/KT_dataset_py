import numpy as np

import pandas as pd 

import json

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

from shapely.geometry import shape, Point, Polygon

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
data_df = pd.read_csv("/kaggle/input/covid19-romania-county-level/ro_covid_19_time_series.csv")
ro_geo_data = "/kaggle/input/elementary-school-admission-romania-2014/romania.geojson"
ro_map = folium.Map(location=[45.9, 24.9], zoom_start=7)



folium.GeoJson(

    ro_geo_data,

    name='geojson'

).add_to(ro_map)



folium.LayerControl().add_to(ro_map)



ro_map
with open(ro_geo_data) as json_file:

    json_data = json.load(json_file)
county_lat_long_df = pd.DataFrame()

for item in json_data['features']:

    polygons = list(shape(item['geometry']))

    county = item['properties']['name']

    #print(county, polygons[0].centroid)

    county_lat_long_df = county_lat_long_df.append(pd.DataFrame({'county': county, 'Lat':polygons[0].centroid.y, 'Long': polygons[0].centroid.x}, index=[0]))
print(list(data_df.County.unique()))
county_join = pd.DataFrame(list(data_df.County.unique()))

county_join.columns = ['County']

county_join = county_join.loc[~(county_join.County=='Not identified')]

county_join.head()
import difflib 

difflib.get_close_matches

county_lat_long_df['County'] = county_lat_long_df.county.map(lambda x: difflib.get_close_matches(x, county_join.County)[0])

print(f"Validation: {county_lat_long_df.County.nunique()},{county_lat_long_df.county.nunique()}")

county_lat_long_df.head()
data_df = data_df.merge(county_lat_long_df, on='County', how='inner')
last_data_df = data_df.loc[data_df.Date==max(data_df.Date)].reset_index()

last_data_df.head(10)

print(f"Validation: {last_data_df.county.nunique()}")
ro_map = folium.Map(location=[45.9, 24.9], zoom_start=6)



folium.Choropleth(

    geo_data=ro_geo_data,

    name='choropleth',

    data=last_data_df,

    columns=['county', 'Confirmed'],

    key_on='feature.properties.name',

    fill_color='Reds',

    fill_opacity=0.6,

    line_opacity=0.5,

    legend_name='Confirmed cases / county'

).add_to(ro_map)







radius_min = 3

radius_max = 50

weight = 1

fill_opacity = 0.2



_color_conf = 'Magenta'

group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed cases</span>')

for i in range(len(last_data_df)):

    lat = last_data_df.loc[i, 'Lat']

    lon = last_data_df.loc[i, 'Long']

    county = last_data_df.loc[i, 'county']



    _radius_conf = np.sqrt(last_data_df.loc[i, 'Confirmed'])

    if _radius_conf < radius_min:

        _radius_conf = radius_min



    if _radius_conf > radius_max:

        _radius_conf = radius_max



    _popup_conf = str(county) + '\nConfirmed: '+str(last_data_df.loc[i, 'Confirmed'])

                                                    

    folium.CircleMarker(location = [lat,lon], 

                        radius = _radius_conf, 

                        popup = _popup_conf, 

                        color = _color_conf, 

                        fill_opacity = fill_opacity,

                        weight = weight, 

                        fill = True, 

                        fillColor = _color_conf).add_to(group0)



group0.add_to(ro_map)





folium.LayerControl().add_to(ro_map)



ro_map

    
def plot_count(feature, value, title, df, size=4):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    df = df.sort_values([value], ascending=False).reset_index(drop=True)

    g = sns.barplot(df[feature], df[value], palette='Set3')

    g.set_title("Number of {}".format(title))

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

    plt.show()  
plot_count('county', 'Confirmed', 'Confirmed cases per county (Judet) - last day', last_data_df)
def plot_time_variation(df, y='Confirmed', hue='county', size=1, is_log=False):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    g = sns.lineplot(x="Date", y=y, hue=hue, data=df)

    plt.xticks(rotation=90)

    plt.title(f'{y} cases grouped by {hue}')

    if(is_log):

        ax.set(yscale="log")

        plt.title(f'{y} cases grouped by {hue} - log scale')

    else:

        plt.title(f'{y} cases grouped by {hue}')

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()  
plot_time_variation(data_df, size=5, is_log=True)