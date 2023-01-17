import pandas as pd
import numpy as np
import os

from datetime import datetime
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import plotly
import seaborn as sns
import folium

import json
data_sf_fire = pd.read_csv('../input/sf-fire-data-incidents-violations-and-more/fire-department-calls-for-service.csv')
date_columns = ['Call Date','Watch Date','Received DtTm','Entry DtTm','Dispatch DtTm',
                'Response DtTm','On Scene DtTm','Transport DtTm','Hospital DtTm',
                'Available DtTm']

for c in date_columns:
    data_sf_fire[c] = pd.to_datetime(data_sf_fire[c])

data_start_date = np.max(data_sf_fire['Call Date']) - relativedelta(months=1)
data_sf_fire_recent = data_sf_fire[data_sf_fire['Call Date']>=data_start_date]
average_entry_time = np.mean(data_sf_fire_recent['Entry DtTm'] - data_sf_fire_recent['Received DtTm']).total_seconds()
print('Average time taken to enter details of an incident (in minutes):',round(average_entry_time/60,2))

average_dispatch_time = np.mean(data_sf_fire_recent['Dispatch DtTm'] - data_sf_fire_recent['Entry DtTm']).total_seconds()
print('Average time taken to dispatch to the appropriate team (in minutes):',round(average_dispatch_time/60,2))

average_response_time = np.mean(data_sf_fire_recent['Response DtTm'] - data_sf_fire_recent['Dispatch DtTm']).total_seconds()
print('Average time taken by the team to respond to the incident (in minutes):',round(average_response_time/60,2))

average_on_scene_time = np.mean(data_sf_fire_recent['On Scene DtTm'] - data_sf_fire_recent['Response DtTm']).total_seconds()
print('Average time taken by the team to arrive on scene of the incident (in minutes):',round(average_on_scene_time/60,2))
number_of_incidents_by_zipcode = data_sf_fire_recent.groupby(['Zipcode of Incident']).size().reset_index()
number_of_incidents_by_zipcode.columns = ['zipcode','count']
number_of_incidents_by_zipcode['zipcode'] = number_of_incidents_by_zipcode['zipcode'].astype(int)
number_of_incidents_by_zipcode['zipcode'] = number_of_incidents_by_zipcode['zipcode'].astype(str)

popup_dict = {'94121':[37.7802,-122.4938],
              '94122':[37.7623,-122.4897],
              '94116':[37.7444,-122.4842],
              '94132':[37.7213,-122.4890],
              '94118':[37.7810,-122.4615],
              '94117':[37.7715,-122.4433],
              '94114':[37.7585,-122.4364],
              '94131':[37.7447,-122.4437],
              '94127':[37.7365,-122.4594],
              '94112':[37.7191,-122.4450],
              '94123':[37.8003,-122.4368],
              '94115':[37.7862,-122.4371],
              '94109':[37.7913,-122.4213],
              '94102':[37.7794,-122.4220],
              '94103':[37.7742,-122.4107],
              '94110':[37.7506,-122.4162],
              '94134':[37.7186,-122.4134],
              '94124':[37.7341,-122.3911],
              '94107':[37.7577,-122.3963],
              '94105':[37.7897,-122.3939],
              '94111':[37.7957,-122.4001],
              '94108':[37.7922,-122.4086],
              '94133':[37.8025,-122.4107],
              '94129':[37.7984,-122.4663],
              '94130':[37.8231,-122.3705],
              '94104':[37.7914,-122.4015]
             }

popup_coords = pd.DataFrame(popup_dict).T.reset_index()
popup_coords.columns = ['zipcode','lat','lon']

map_data_set = pd.merge(number_of_incidents_by_zipcode,popup_coords,how='left')
map_data_set
district_geo = r'../input/sf-json/comprehensive_sf_geojson.json'
geo_json_data = json.load(open(district_geo))
sf_coordinates = (37.76, -122.45)

map1 = folium.Map(location=sf_coordinates, zoom_start=12)

for i in map_data_set.index:
    try:
        folium.Marker(location=[map_data_set.iloc[i]['lat'],map_data_set.iloc[i]['lon']],
                      popup='No. of incidents: {}'.format(map_data_set.iloc[i]['count'])).add_to(map1)
    except ValueError:
        pass

folium.Choropleth(geo_data=district_geo,
                  legend_name = 'Number of incidents by Zipcode',
                  data = map_data_set,
                  columns = ['zipcode', 'count'],
                  key_on = 'feature.properties.ZIP',
                  fill_color = 'YlOrRd',
                  fill_opacity = 0.7,
                  line_opacity = 0.2,
                  highlight = True
               ).add_to(map1)

display(map1)