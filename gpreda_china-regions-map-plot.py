import numpy as np

import pandas as pd

import os

import folium
all_data_folder = "/kaggle/input"

print(os.listdir(all_data_folder))

cn_data_folder = "/kaggle/input/china-regions-map/"

print(os.listdir(cn_data_folder))

wd_data_folder = "/kaggle/input/python-folio-country-boundaries"

print(os.listdir(wd_data_folder))

regions_geo = os.path.join(cn_data_folder,"china.json")

wd_geo = os.path.join(wd_data_folder,"world-countries.json")
fullJsonString = None

with open(regions_geo, 'r') as f:

    fullJsonString = f.read().encode('ascii', errors='ignore')
ch_map = folium.Map(location=[35, 100], zoom_start=4)



folium.Choropleth(

    regions_geo,

    name='China regions',

    fill_opacity=0.1,

    line_opacity=0.2,

).add_to(ch_map)



folium.LayerControl().add_to(ch_map)



ch_map
wd_map = folium.Map(location=[0, 0], zoom_start=2)



folium.Choropleth(

    geo_data=wd_geo,

    name='Countries of the World',

    fill_opacity=0.2,

    line_opacity=0.2,



).add_to(wd_map)





folium.LayerControl().add_to(wd_map)



wd_map