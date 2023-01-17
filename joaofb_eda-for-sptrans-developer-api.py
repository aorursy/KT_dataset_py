import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import pickle

import folium

from folium.plugins import HeatMap

import sqlite3

import plotly.express as px

import os



pd.options.display.max_columns = None

pd.options.display.max_rows = 300
data_dir = os.path.abspath('../input/so-paulo-bus-system/')
overview = pd.read_csv(os.path.join(data_dir, 'overview.csv'))

overview.head()
overview.info()
trips = pd.read_csv(os.path.join(data_dir, 'trips.csv'))
trips.head()
trips.info()
routes = pd.read_csv(os.path.join(data_dir, 'routes.csv'))

routes.head()
len(routes['route_id'].unique())
stops = pd.read_csv(os.path.join(data_dir, 'stops.csv'))

stops.head()
stops.info()
import numpy as np
stops['stop_desc'] = stops['stop_desc'].apply(lambda x: x if x != None else np.nan)
len(stops['stop_id'].unique())
def generate_base_map(default_location=[-23.5489, -46.6388],default_zoom_start=11,):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map
folium_map = generate_base_map()
# Getting unique stops

unique_stops = stops.drop_duplicates(['stop_id'])



# Generating map

for i in range(len(unique_stops)):

    marker = folium.CircleMarker(location=[unique_stops['stop_lat'][i], unique_stops['stop_lon'][i]], radius = 1, color='r', fill=True)

    marker.add_to(folium_map)
folium_map
stops['count'] = 1

base_map = generate_base_map()

HeatMap(data=stops[['stop_lat', 'stop_lon', 'count']].groupby(['stop_lat', 'stop_lon']).sum().reset_index().values.tolist(), radius=8, max_zoom=15).add_to(base_map)

base_map
shapes = pd.read_csv(os.path.join(data_dir, 'shapes.csv'))
shapes.shape
shapes.head()
shapes['shape_coords'] = shapes.apply(lambda x: (x['shape_pt_lat'], x['shape_pt_lon']), axis=1)
shapes['shape_coords'].head()
import random

def random_color():

    a = random.randint(0,256)

    b = random.randint(0,256)

    c = random.randint(0,256)

    rgbl=[a,b,c]

    random.shuffle(rgbl)

    return tuple(rgbl)



def genhex():

    rgb = random_color()

    return '#%02x%02x%02x' % rgb
folium_map = generate_base_map()
for shape in list(shapes.groupby('shape_id')):

    df = shape[1]

    marker = folium.PolyLine(locations=df['shape_coords'].to_list(), color=genhex())

    marker.add_to(folium_map)
'''

for shape in list(shapes.groupby('shape_id')):

    df = shape[1]

    coord_list = df['shape_coords'].to_list()

    initial_point = coord_list[0]

    terminal_point = coord_list[len(coord_list)-1]

    route_edges = [initial_point, terminal_point]

    

    for point in route_edges:

        marker = folium.Marker(location=[point[0], point[1]])

        marker.add_to(folium_map)

'''

0
folium_map
overview.tail(80)
stops_quantity = overview.groupby('trip_id').count()

stops_quantity['index'].describe()
stops_quantity.rename(columns = {'index':'stops_quantity'}, inplace=True)
px.histogram(stops_quantity, x='stops_quantity', histnorm='density', labels={'stops_quantity':'Number of Stops'})
stops_quantity.reset_index(inplace=True)
stops_quantity.head()
stops_quantity_merger = stops_quantity[['trip_id','stops_quantity']].copy()
overview_stops_qnt = overview.merge(stops_quantity_merger, on='trip_id', how='outer')
shapes_stops_qnt = shapes.merge(overview_stops_qnt.drop_duplicates(['shape_id']), on='shape_id', how='outer')
shapes_stops_qnt.drop(['index_x', 'index_y'], axis=1, inplace=True)
many_stops_shapes = shapes_stops_qnt[shapes_stops_qnt['stops_quantity'] >= 80]
folium_map = generate_base_map()
for shape in list(many_stops_shapes.groupby('shape_id')):

    df = shape[1]

    marker = folium.PolyLine(locations=df['shape_coords'].to_list(), color=genhex())

    marker.add_to(folium_map)
folium_map
few_stops_shapes = shapes_stops_qnt[shapes_stops_qnt['stops_quantity'] <= 10]
folium_map = generate_base_map()
for shape in list(few_stops_shapes.groupby('shape_id')):

    df = shape[1]

    marker = folium.PolyLine(locations=df['shape_coords'].to_list(), color=genhex())

    marker.add_to(folium_map)
folium_map
overview.head()
overview.corr()
stop_times = pd.read_csv(os.path.join(data_dir, 'stop_times.csv'))
stop_times.head()
frequencies = pd.read_csv(os.path.join(data_dir, 'frequencies.csv'))
frequencies.head()
frequencies[frequencies['trip_id'] == '1012-10-0']
overview.head()
passengers = pd.read_csv(os.path.join(data_dir, 'passengers.csv'))

passengers.head()
# bus_position = pd.read_csv(os.path.join(data_dir, 'bus_position.csv'))

# bus_position.head()
# bus_position.tail()