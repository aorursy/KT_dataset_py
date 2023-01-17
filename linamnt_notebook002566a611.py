import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # simple plots

import networkx as nx # network analysis
trips = pd.read_csv("../input/trip.csv", header=0)

stations = pd.read_csv("../input/station.csv", header=0)

 

stations_sf = stations[stations.city == 'San Francisco']

trips = trips[trips.start_station_id ]

#trips.slat = pd.merge(trips, stations, )
trips.start_station_id in list(stations_sf.id)