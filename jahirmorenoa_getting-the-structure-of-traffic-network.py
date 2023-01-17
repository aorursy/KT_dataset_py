# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
traffic_df = pd.read_csv('/kaggle/input/mock-traffic-data/MockTrafficDataForMCNFP.csv')

# Make the columns that are dates into actual dates so you can do arithmetic on them

traffic_df = traffic_df.set_index('id')

cols = ['time_node_1', 'time_node_2', 'time_node_3','time_node_4', 'time_node_5', 'time_node_6']



for col in cols:

    traffic_df[col] = pd.to_datetime(traffic_df[col]).dt.time



traffic_df.head()
import re

import networkx as nx



def get_graph(traffic_df):



    path_nodes = [re.findall('\d', traffic_df.iloc[i].dropna().index[1:][j])[0] for i in range(len(traffic_df)) for j in range(len(traffic_df.iloc[i].dropna().index[1:]))]



    path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1) if path_nodes[i]!='6']

    

    cars_per_edge = [path_edges.count(i) for i in set(path_edges)]

        

    edges = list(set(path_edges))

    

    path =[]

    path_travel=[]

    for i in path_nodes:

        if i!='6':

            path.append(i)

        else:

            path.append(i)

            path_travel.append(tuple(path))

            path = []

    

    G = nx.DiGraph()

    

    for edge, weight in zip(edges, cars_per_edge):

        G.add_edges_from([edge], num_cars=weight)

    return G, path_travel, path_edges
import matplotlib.pyplot as plt



%matplotlib inline

plt.figure(figsize=(10,9))



G, path_travel, path_edges = get_graph(traffic_df)



edge_width = [0.01*G[u][v]['num_cars'] for u,v in G.edges()]



pos = {'1':(-0.6,0), '2':(-0.4,0.4), '3':(-0.4, -0.4), '4':(0.3,0.4), '5':(0.3, -0.4), '6':(0.7, 0)}

nx.draw_networkx(G, pos=pos, alpha=0.7, width=edge_width, node_size=1500, edge_color='.4', cmap=plt.cm.Blues)

labels = nx.draw_networkx_edge_labels(G, pos=pos, font_size=10)

plt.title('How many cars a path is travel from 7am to 7pm', size=20)

plt.show()
from datetime import timedelta

end_time = [timedelta(hours=x.hour, minutes=x.minute, seconds=x.second) for x in traffic_df['time_node_6']]

start_time = [timedelta(hours=x.hour, minutes=x.minute, seconds=x.second) for x in traffic_df['time_node_1']]



eda_traffic = traffic_df.copy()

eda_traffic['avg_time_travel'] = np.array(end_time) - np.array(start_time)

eda_traffic['path_travel'] = path_travel

time_travel_paths = pd.DataFrame(eda_traffic.groupby(by='path_travel')['avg_time_travel'].agg(lambda x: x.mean()))

ranking_time_travel_nodes = time_travel_paths.reset_index()

ranking_time_travel_nodes
ranking_time_travel_nodes.iloc[0]['path_travel']
time_travel_paths['avg_time_travel'] = time_travel_paths['avg_time_travel'].apply(lambda x : x.total_seconds()/3600)

plt.figure(figsize=(10,9))

r_bins = range(len(time_travel_paths))

plt.bar(r_bins, time_travel_paths['avg_time_travel'], color='g')

plt.xticks(r_bins, list(time_travel_paths.index), rotation=30)

plt.yticks()

plt.xlabel('Traveling Paths throgh nodes')

plt.ylabel('Amount of time traveling (hours)')

plt.title('Path traveling time')

plt.show()