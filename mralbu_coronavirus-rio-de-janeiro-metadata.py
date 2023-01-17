import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import geopandas as gpd

import folium



coronavirus_rj_case_metadata = pd.read_csv('../input/coronavirus-rio-de-janeiro/coronavirus_rj_case_metadata.csv')

spatial_rj_neighborhoods = gpd.read_file('../input/coronavirus-rio-de-janeiro/spatial_rj_neighborhoods.gpkg')
coronavirus_rj_case_metadata.info()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))



coronavirus_rj_case_metadata.faixa_etária.value_counts().sort_index()[:-1].plot(kind='barh', ax=ax1, title='Coronavirus RJ Cases by Age', xlim = (0, 600));

coronavirus_rj_case_metadata.sexo.value_counts().sort_index()[:-1].plot(kind='barh', title='Coronavirus RJ Cases by Sex', xlim = (0, 1400));



for p in ax1.patches:

    ax1.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()), xytext=(2, 4), textcoords='offset points')

for p in ax2.patches:

    ax2.annotate("%.0f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y()), xytext=(2, 25), textcoords='offset points')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))



coronavirus_rj_case_metadata.hospitalizações.value_counts(normalize=True).plot(kind='bar', ylim=(0, 1), title='Hospitalized?', ax=ax1)

coronavirus_rj_case_metadata.uti.value_counts(normalize=True).plot(kind='bar', ylim=(0, 1), title='Intensive Care Unit?', ax=ax2)



for ax in [ax1, ax2]:

    for p in ax.patches:

        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))



(coronavirus_rj_case_metadata [coronavirus_rj_case_metadata.tp_gestao.isin(['PUBLICA', 'PARTICULAR'])]

     .tp_gestao.value_counts(normalize=True)

     .plot(kind='bar', ylim=(0, 1), title='Hospital Management', ax=ax1));



(coronavirus_rj_case_metadata [coronavirus_rj_case_metadata.tipo_municipal.isin(['SMS', 'PART', 'SES', 'FED', 'MIL'])]

     .tipo_municipal.value_counts(normalize=True)

     .plot(kind='bar', ylim=(0, 1), title='Hospital Category', ax=ax2));



for ax in [ax1, ax2]:

    for p in ax.patches:

        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
spatial_rj_neighborhoods.info()
spatial_rj_neighborhoods[spatial_rj_neighborhoods.lat < 0].plot('cases', markersize='cases');
coronavirus_rj_map = folium.Map(location = [-22.9, -43.4], zoom_start = 11)



incidents_accident = folium.map.FeatureGroup()

latitudes = list(spatial_rj_neighborhoods[spatial_rj_neighborhoods.lat < 0].lat)

longitudes = list(spatial_rj_neighborhoods[spatial_rj_neighborhoods.lat < 0].lon)

labels = list(spatial_rj_neighborhoods[spatial_rj_neighborhoods.lat < 0].neighborhood)

case_numbers = list(spatial_rj_neighborhoods[spatial_rj_neighborhoods.lat < 0].cases)

death_numbers = list(spatial_rj_neighborhoods[spatial_rj_neighborhoods.lat < 0].deaths)



for lat, lng, label, cases, deaths in zip(latitudes, longitudes, labels, case_numbers, death_numbers):

    folium.Circle(

      location = [lat, lng], 

      tooltip = label + f':\n Casos - {cases}\n    Mortes - {deaths}',

      radius = cases * 10,

      fill= True

     ).add_to(coronavirus_rj_map) 

    

coronavirus_rj_map