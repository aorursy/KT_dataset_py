# IMPORT MODULES



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx

import re

import sys

import warnings

import mplleaflet

import matplotlib.colors

warnings.filterwarnings("ignore")







# LIST INPUT FILES



from subprocess import check_output

print('INPUT FILES\n-------\n{}'.format(check_output(["ls", "../input"]).decode("utf8")))
# Create Data Frames from .CSV files



# Travel history

travel_df_2014 = pd.read_csv('../input/OD_2014.csv')

df = travel_df_2014



# Stations information

stations_df_2014 = pd.read_csv('../input/Stations_2014.csv')

df_stations = stations_df_2014
df.head()
# Create a count matrix

matrix = df.groupby(

    [

        'start_station_code', 

        'end_station_code'

    ]

)['end_station_code'].count().unstack(fill_value=0)



# Print head and Visualize

fig = plt.figure(figsize=(10,10))

sns.clustermap(matrix,vmin = 0, vmax=100,cmap=plt.cm.Reds)



matrix.head().ix[:,:5006]
df_stations.head()
# Create a graph containing all nodes (stations) and edges (travels)



H=nx.Graph()



for v in range(len(df_stations)):

    H.add_node(

        df_stations['name'][v], 

        pos = (

            df_stations['longitude'][v], 

            df_stations['latitude'][v]

        )

    )



for x in range(len(df_stations)):

    for y in range(len(df_stations)):

        if matrix.loc[df_stations['code'][x]].loc[df_stations['code'][y]]:

            H.add_edge(

                df_stations['name'][x],

                df_stations['name'][y],

                weight = matrix.loc[df_stations['code'][x]].loc[df_stations['code'][y]]

                )

# Create a centrality DataFrame



centrality_df = pd.concat([

    pd.DataFrame(

        nx.centrality.closeness_centrality(H), 

        index = ['closeness_centrality']

    ),

    pd.DataFrame(

        nx.centrality.betweenness_centrality(H), 

        index = ['betwenness_centrality']

    ),

    pd.DataFrame(

        nx.centrality.communicability_centrality(H), 

        index = ['communicability_centrality']

    ),

    pd.DataFrame(

        nx.centrality.degree_centrality(H), 

        index = ['degree_centrality']

    ),

    pd.DataFrame(

        nx.centrality.eigenvector_centrality(H), 

        index = ['eigenvector_centrality']

    ),

    pd.DataFrame(

        nx.centrality.harmonic_centrality(H), 

        index = ['harmonic_centrality']

    ),

    pd.DataFrame(

        nx.centrality.information_centrality(H), 

        index = ['information_centrality']

    ),

    pd.DataFrame(

        nx.centrality.katz_centrality_numpy(H), 

        index = ['katz_centrality']

    ),

    pd.DataFrame(

        nx.centrality.load_centrality(H), 

        index = ['load_centrality']

    )

])
# Create a function to plot the network. 

# It accepts several arguments to personalize.



def bixi_network(node_threshold = 18000, edge_threshold = 200, node_pos = None, 

                 centrality = 'degree',scale=1,colormap = plt.cm.BrBG_r, vmin = 0,

                 vmax = 1, alpha = 1):

    

    # Keep nodes who pass the threshold

    new_df_stations = df_stations[

        np.array(

            list(

                map(

                    lambda x: np.sum(matrix[df_stations['code'][x]].dropna().values),

                    range(len(df_stations))

                )

            )

        ) > node_threshold].reset_index(drop=True)

    

    # Create a graph with nodes

    H=nx.Graph()

    for v in range(len(new_df_stations)):

        H.add_node(

            new_df_stations['name'][v], 

            pos = (

                new_df_stations['longitude'][v], 

                new_df_stations['latitude'][v],

            ),

            label = '' # By default, no names

        )

        

    # Loop inside the DataFrame and create edges for every travels  

    for x in range(len(new_df_stations)):

        for y in range(len(new_df_stations)):

            if matrix.loc[new_df_stations['code'][y]].loc[new_df_stations['code'][x]] > edge_threshold:

                H.add_edge(

                    new_df_stations['name'][x],

                    new_df_stations['name'][y],

                    weight = matrix.loc[new_df_stations['code'][x]].loc[new_df_stations['code'][y]]

                )

    

    pos = nx.get_node_attributes(H,'pos')

    edges = H.edges()

    weights = [H[u][v]['weight'] for u,v in edges]

    

    if node_pos == 'spring_layout':

        pos = nx.spring_layout(H, k=1)

    

    # Adjust node size by total count

    size = list(

        map(

            lambda x: np.sum(matrix[new_df_stations['code'][x]].dropna().values),

            range(len(new_df_stations))

        )

    )

    

    # Create the scale from 1 to 9

    size = np.array(size)**float('1.{}'.format(scale))

    

    # New version of centrality_df with the concerned nodes

    new_centrality_df = centrality_df[H.nodes()]

    

    # Create labels only for the nodes with the best and worst centrality scores

    less = list(

        new_centrality_df.loc[

            '{}_centrality'.format(centrality)

        ].sort_values(

            ascending = True

        ).head().index[:3]

    )

    

    for x in range(len(less)):

        H.node[less[x]]['label'] = '{} : {}'.format(

            less[x], 

            new_centrality_df.loc[

                '{}_centrality'.format(centrality)

            ].loc[less[x]]

        )

    

    top = list(

        new_centrality_df.loc[

            '{}_centrality'.format(centrality)

        ].sort_values(

            ascending = False

        ).head().index[:3]

    )



    for x in range(len(top)):

        H.node[top[x]]['label'] = '{} : {}'.format(

            top[x], 

            round(new_centrality_df.loc[

                '{}_centrality'.format(centrality)

            ].loc[top[x]],2)

        )

    

    labels = nx.get_node_attributes(H,'label')

    

    # Draw the graph

    nx.draw(H, 

            pos, 

            node_color = list(new_centrality_df.loc['{}_centrality'.format(centrality)]),

            cmap = colormap, 

            node_size=(np.array(size)/300), 

            alpha = alpha,

            vmin = vmin,

            vmax = vmax,

            width = np.array(weights)/np.array(weights).max(),

            labels = labels)

    

    # Normalize the color scale and create a colorbar

    cmap = colormap

    norm = matplotlib.colors.Normalize(

        vmin = vmin, 

        vmax = vmax

    )

    sm = plt.cm.ScalarMappable(

        cmap=cmap, 

        norm=norm

    )

    

    sm.set_array([])

    fig.colorbar(sm)
# Plot degree centrality



fig = plt.figure(figsize=(14,14))



bixi_network(

    node_threshold = 14000, 

    edge_threshold = 100, 

    node_pos = None, 

    scale = 3,

    centrality = 'degree',

    colormap = plt.cm.Greens,

    vmin = 0.90,

    vmax = 0.98, 

    alpha = 0.8)



plt.title('degree_centrality')
fig = plt.figure(figsize=(14,14))



bixi_network(

    node_threshold = 14000, 

    edge_threshold = 100, 

    node_pos = None, 

    scale = 3,

    centrality = 'eigenvector',

    colormap = plt.cm.Reds,

    vmin = 0,

    vmax = 0.25, 

    alpha = 0.8)

    



plt.title('eigenvector_centrality')
# For fancier maping, we can wrap it inside mplleaflet



mplleaflet.display(

    bixi_network(

        node_threshold = 20000, 

        edge_threshold = 10, 

        node_pos = None, 

        scale = 2,

        centrality = 'degree',

        colormap = plt.cm.Greens,

        vmin = 0.90,

        vmax = 0.98, 

        alpha = 0.75),

    tiles = 'cartodb_positron')