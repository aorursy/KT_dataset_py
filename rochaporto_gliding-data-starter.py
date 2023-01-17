import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
flight_websource = pd.read_csv("../input/gliding-data/flight_websource.csv")
flight_track = pd.read_csv("../input/gliding-data/flight_track.csv")
flight_phases = pd.read_csv("../input/gliding-data/phases.csv", skiprows=lambda i: i>0 and random.random() > 0.5)
flight_websource.head(1)
flight_track.head(1)
flight_websource.groupby(['Country', 'Region'])['Region'].value_counts().sort_values(ascending=False).head(3)
flight_websource.groupby(['Country', 'Year', 'Takeoff'])['Takeoff'].value_counts().sort_values(ascending=False).head(3)
flight_websource['DayOfWeek'] = flight_websource.apply(lambda r: datetime.datetime.strptime(r['Date'], "%Y-%m-%dT%H:%M:%SZ").strftime("%A"), axis=1)
flight_websource.groupby(['DayOfWeek'])['DayOfWeek'].count().plot.bar()
flight_websource['Month'] = flight_websource.apply(lambda r: datetime.datetime.strptime(r['Date'], "%Y-%m-%dT%H:%M:%SZ").strftime("%m"), axis=1)
flight_websource.groupby(['Month'])['Month'].count().plot.bar()
flight_all = pd.merge(flight_websource, flight_track, how='left', on='ID')
flight_all.head(1)
flight_phases.head(1)
phases = pd.merge(flight_phases, flight_websource[['TrackID', 'Distance', 'Speed']], on='TrackID')
phases['Lat'] = np.rad2deg(phases['CentroidLatitude'])
phases['Lng'] = np.rad2deg(phases['CentroidLongitude'])

phases_copy = phases[phases.Type==5][phases.AvgVario<10][phases.AvgVario>2].copy()
phases_copy.head(2)

#phases_copy['AM'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%p"), axis=1)
#phases_copy['Day'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%j"), axis=1)
#phases_copy['Week'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%W"), axis=1)
#phases_copy['Month'] = phases_copy.apply(lambda r: r['StartTime'][5:7], axis=1)
#phases_copy['Year'] = phases_copy.apply(lambda r: r['StartTime'][0:4], axis=1)
#phases_copy['YearMonth'] = phases_copy.apply(lambda r: r['StartTime'][0:7], axis=1)
#phases_copy['YearMonthDay'] = phases_copy.apply(lambda r: r['StartTime'][0:10], axis=1)

# use the corresponding function above to update the grouping to something other than week
phases_copy['Group'] = phases_copy.apply(lambda r: datetime.datetime.strptime(r['StartTime'], "%Y-%m-%dT%H:%M:%SZ").strftime("%W"), axis=1)
phases_copy.head(1)
# This is a workaround for this known issue:
# https://github.com/python-visualization/folium/issues/812#issuecomment-582213307
!pip install git+https://github.com/python-visualization/branca
!pip install git+https://github.com/sknzl/folium@update-css-url-to-https
import folium
from folium import plugins
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, HeatMapWithTime, MarkerCluster

# folium.__version__ # should be '0.10.1+8.g4ea1307'
# folium.branca.__version__ # should be '0.4.0+4.g6ac241a'
# we use a smaller sample to improve the visualization
# a better alternative is to group entries by CellID, an example of this will be added later
phases_single = phases_copy.sample(frac=0.01, random_state=1)
m_5 = folium.Map(location=[47.06318, 5.41938], tiles='stamen terrain', zoom_start=7)
HeatMap(
    phases_single[['Lat','Lng','AvgVario']], gradient={0.5: 'blue',  0.7:  'yellow', 1: 'red'},
    min_opacity=5, max_val=phases_single.AvgVario.max(), radius=4, max_zoom=7, blur=4, use_local_extrema=False).add_to(m_5)

m_5
m_5 = folium.Map(location=[47.06318, 5.41938], tiles='stamen terrain', zoom_start=7)

groups = phases_copy.Group.sort_values().unique()
data = []
for g in groups:
    data.append(phases_copy.loc[phases_copy.Group==g,['Group','Lat','Lng','AvgVario']].groupby(['Lat','Lng']).sum().reset_index().values.tolist())
    
HeatMapWithTime(
    data,
    index = list(phases_copy.Group.sort_values().unique()),
    gradient={0.1: 'blue',  0.3:  'yellow', 0.8: 'red'},
    auto_play=True, scale_radius=False, display_index=True, radius=4, min_speed=1, max_speed=6, speed_step=1,
    min_opacity=1, max_opacity=phases_copy.AvgVario.max(), use_local_extrema=True).add_to(m_5)

m_5
