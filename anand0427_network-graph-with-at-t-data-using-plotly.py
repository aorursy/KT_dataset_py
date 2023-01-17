import pandas as pd

import numpy as np

import networkx as nx

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import plotly.graph_objs as go
init_notebook_mode(connected=True)
network_df = pd.read_csv("../input/network_data.csv") 
network_df.head()
A = list(network_df["source_ip"].unique())
B = list(network_df["destination_ip"].unique())
node_list = set(A+B)
G = nx.Graph()
for i in node_list:

    G.add_node(i)
#G.nodes()
for i,j in network_df.iterrows():

    G.add_edges_from([(j["source_ip"],j["destination_ip"])])
pos = nx.spring_layout(G, k=0.5, iterations=50)
#pos
for n, p in pos.items():

    G.node[n]['pos'] = p
edge_trace = go.Scatter(

    x=[],

    y=[],

    line=dict(width=0.5,color='#888'),

    hoverinfo='none',

    mode='lines')



for edge in G.edges():

    x0, y0 = G.node[edge[0]]['pos']

    x1, y1 = G.node[edge[1]]['pos']

    edge_trace['x'] += tuple([x0, x1, None])

    edge_trace['y'] += tuple([y0, y1, None])
node_trace = go.Scatter(

    x=[],

    y=[],

    text=[],

    mode='markers',

    hoverinfo='text',

    marker=dict(

        showscale=True,

        colorscale='RdBu',

        reversescale=True,

        color=[],

        size=15,

        colorbar=dict(

            thickness=10,

            title='Node Connections',

            xanchor='left',

            titleside='right'

        ),

        line=dict(width=0)))



for node in G.nodes():

    x, y = G.node[node]['pos']

    node_trace['x'] += tuple([x])

    node_trace['y'] += tuple([y])
for node, adjacencies in enumerate(G.adjacency()):

    node_trace['marker']['color']+=tuple([len(adjacencies[1])])

    node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))

    node_trace['text']+=tuple([node_info])
fig = go.Figure(data=[edge_trace, node_trace],

             layout=go.Layout(

                title='<br>AT&T network connections',

                titlefont=dict(size=16),

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text="No. of connections",

                    showarrow=False,

                    xref="paper", yref="paper") ],

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))



iplot(fig)

plotly.plot(fig)
from plotly.plotly import plot
from plotly import plotly
import plotly
plotly.tools.set_credentials_file(username='anand0427', api_key='5Xd8TlYYqnpPY5pkdGll')
iplot(fig,"anand0427",filename="Network Graph.html")