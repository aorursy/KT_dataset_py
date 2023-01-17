# Install required packages

# (don't forget to enable internet access !)

!apt-get -y install python-rtree

!pip install osmnx
import numpy as np

import pandas as pd 

import os

import osmnx as ox

import networkx as nx

print(os.listdir("../input"))
# Get a part of the train data

train = pd.read_csv('../input/train.csv', nrows=10)

print(train)
# Get shape of NewYork City from OSM website

NYC = ox.gdf_from_place('New York city, US') 

# Get streets from NewYork City from OSM website

NY_streets = ox.graph_from_place('New York city, US')

ox.save_graphml(NY_streets, 'nyc.graphml', folder='../geodata/')
# Load data from graphml (gives better performances on graph computing)

NY_streets = ox.load_graphml('nyc.graphml', folder='../geodata/')
# Do what you want now...

# Example : compute driving distance

for index, row in train.iterrows():

    # Finds the nearest node of the graph from the point given (lat, lng)  

    pickup, p_add = ox.get_nearest_node(NY_streets, (row['pickup_latitude'], row['pickup_longitude']), method='haversine', return_dist=True)

    dropoff, d_add = ox.get_nearest_node(NY_streets, (row['dropoff_latitude'], row['dropoff_longitude']), method='haversine', return_dist=True)

    print(nx.shortest_path_length(NY_streets, source=pickup, target=dropoff, weight='length') + p_add + d_add)