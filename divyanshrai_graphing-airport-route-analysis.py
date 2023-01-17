%%capture

!pip install networkx==2.3
import pandas as pd

import numpy as np

import networkx as nx

#%matplotlib notebook

import matplotlib.pyplot as plt

import os

import operator

import warnings

warnings.filterwarnings('ignore')

import plotly.graph_objects as go

print(os.listdir("../input"))
cols_list=["Source airport","Destination airport"]

G_df = pd.read_csv('../input/openflights-route-database-2014/routes.csv',usecols=cols_list)

#Count_df = pd.read_csv('D:/Personal and studies/College/Semester 6/Social and information networks project/countries.txt')

cols_list=["City","Country","IATA","Latitude","Longitude"]

airport_df = pd.read_csv('../input/openflights-airports-database-2017/airports.csv',usecols=cols_list)
G_df.head(2)
#Count_df.head(2)
airport_df.head(2)
G = nx.from_pandas_edgelist(G_df.head(10000), 'Source airport', 'Destination airport',create_using=nx.DiGraph())
available_IATA=list(airport_df["IATA"])

nodes_in_G=list(G.nodes())
#for i in nodes_in_G:

#    if i not in available_IATA:

#        G.remove_node(i)

#nodes_in_G=list(G.nodes())
dict_pos={}

c=0

for i in nodes_in_G:

    x=np.array(airport_df.loc[airport_df['IATA'].isin([i])][["Latitude","Longitude"]])

    if len(x)==0:

        G.remove_node(i)

        c=c+1

        continue

    else:

        dict_pos[i]=x[0]

print(c)
print(nx.info(G))
nx.set_node_attributes(G, dict_pos, 'pos')
edge_x = []

edge_y = []

for edge in G.edges():

    y0, x0 = G.nodes[edge[0]]['pos']

    y1, x1 = G.nodes[edge[1]]['pos']

    edge_x.append(x0)

    edge_x.append(x1)

    edge_x.append(None)

    edge_y.append(y0)

    edge_y.append(y1)

    edge_y.append(None)



edge_trace = go.Scatter(

    x=edge_x, y=edge_y,

    line=dict(width=0.25, color='#888'),

    hoverinfo='none',

    mode='lines')



node_x = []

node_y = []

for node in G.nodes():

    y, x = G.nodes[node]['pos']

    node_x.append(x)

    node_y.append(y)



node_trace = go.Scatter(

    x=node_x, y=node_y,

    mode='markers',

    hoverinfo='text',

    marker=dict(

        showscale=True,

        colorscale='YlGnBu',

        reversescale=True,

        color=[],

        size=7.5,

        colorbar=dict(

            thickness=15,

            title='Node Connections',

            xanchor='left',

            titleside='right'

        ),

        line_width=2))
node_adjacencies = []

node_text = []

for node, adjacencies in enumerate(G.adjacency()):

    node_adjacencies.append(len(adjacencies[1]))

    node_text.append('# of connections: '+str(len(adjacencies[1]))+' '+np.array(airport_df.loc[airport_df['IATA'].isin([adjacencies[0]])]["Country"])[0]+' , '+np.array(airport_df.loc[airport_df['IATA'].isin([adjacencies[0]])]["City"])[0])



node_trace.marker.color = node_adjacencies

node_trace.text = node_text
node_text[:5]
fig = go.Figure(data=[edge_trace, node_trace],

             layout=go.Layout(

                title='<br>Network graph of airport routes',

                titlefont_size=16,

                showlegend=False,

                hovermode='closest',

                margin=dict(b=10,l=5,r=5,t=10),

                annotations=[ dict(

                    text="",

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002 ) ],

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

                )

fig.show()