import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

%matplotlib inline

DATA_FOLDER = "/kaggle/input/novel-corona-virus-2019-dataset"

GEO_DATA = "/kaggle/input/china-regions-map"

CN_GEO_DATA = '/kaggle/input/coronavirus-latlon-dataset'

data_df = pd.read_csv(os.path.join(DATA_FOLDER, "2019_nCoV_data.csv"))

cn_geo_data = os.path.join(GEO_DATA, "china.json")

cn_ll_data = pd.read_csv(os.path.join(CN_GEO_DATA,"coronavirus_cleaned_21Jan2Feb.csv"))

cn_ll_df = cn_ll_data[['Province/State', 'lat', 'lon']]

cn_ll_df = cn_ll_df.drop_duplicates()

data_cn = data_df.loc[data_df['Country']=="Mainland China"]

data_cn = pd.DataFrame(data_cn.groupby(['Province/State', 'Last Update'])['Confirmed', 'Recovered', 'Deaths'].sum()).reset_index()

data_cn.columns = ['Province/State', 'Update', 'Confirmed', 'Recovered', 'Deaths' ]

data_cn = data_cn.sort_values(by = ['Province/State','Update'], ascending=False)

filtered_data_last = data_cn.drop_duplicates(subset = ['Province/State'],keep='first')

filtered_data_last = filtered_data_last.merge(cn_ll_df, on=['Province/State'])

dt_string = dt.datetime.now().strftime("%d/%m/%Y")

print(f"Kernel last updated: {dt_string}")
m = folium.Map(location=[30, 100], zoom_start=4)

folium.Choropleth(

    geo_data=cn_geo_data,

    name='Confirmed cases - regions',

    key_on='feature.properties.name',

    fill_color='YlGn',

    fill_opacity=0.05,

    line_opacity=0.3,

).add_to(m)



radius_min = 2

radius_max = 40

weight = 1

fill_opacity = 0.2



_color_conf = 'red'

group0 = folium.FeatureGroup(name='Confirmed cases')

for i in range(len(filtered_data_last)):

    lat = filtered_data_last.loc[i, 'lat']

    lon = filtered_data_last.loc[i, 'lon']

    province = filtered_data_last.loc[i, 'Province/State']

    recovered = filtered_data_last.loc[i, 'Recovered']

    death = filtered_data_last.loc[i, 'Deaths']



    _radius_conf = np.sqrt(filtered_data_last.loc[i, 'Confirmed'])

    if _radius_conf < radius_min:

        _radius_conf = radius_min



    if _radius_conf > radius_max:

        _radius_conf = radius_max



    _popup_conf = str(province) + '\n(Confirmed='+str(filtered_data_last.loc[i, 'Confirmed']) + '\nDeaths=' + str(death) + '\nRecovered=' + str(recovered) + ')'

    folium.CircleMarker(location = [lat,lon], 

                        radius = _radius_conf, 

                        popup = _popup_conf, 

                        color = _color_conf, 

                        fill_opacity = fill_opacity,

                        weight = weight, 

                        fill = True, 

                        fillColor = _color_conf).add_to(group0)



group0.add_to(m)

folium.LayerControl().add_to(m)

m