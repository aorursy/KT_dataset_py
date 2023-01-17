import plotly.plotly as py

import plotly.graph_objs as go

import plotly

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

import networkx as nx



G=nx.random_geometric_graph(200,0.125)

pos=nx.get_node_attributes(G,'pos')



dmin=1

ncenter=0

for n in pos:

    x,y=pos[n]

    d=(x-0.5)**2+(y-0.5)**2

    if d<dmin:

        ncenter=n

        dmin=d



p=nx.single_source_shortest_path_length(G,ncenter)
import numpy as np

si = np.random.rand(200) * 30

print(si.shape)
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

        # colorscale options

        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |

        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |

        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |

        colorscale='YlGnBu',

        reversescale=True,

        color=[],

        size= si,

        colorbar=dict(

            thickness=15,

            title='Node Connections',

            xanchor='left',

            titleside='right'

        ),

        line=dict(width=2)))



for node in G.nodes():

    x, y = G.node[node]['pos']

    node_trace['x'] += tuple([x])

    node_trace['y'] += tuple([y])
for node, adjacencies in enumerate(G.adjacency()):

    node_trace['marker']['color']+=tuple([len(adjacencies[1])])

    node_info = '# of connections: '+str(len(adjacencies[1]))+' <br>other data are: <br>File1 <br>File2'

    node_trace['text']+=tuple([node_info])
fig = go.Figure(data=[edge_trace, node_trace],

             layout=go.Layout(

                title='<br>Network graph made with Python',

                titlefont=dict(size=16),

                showlegend=False,

                hovermode='closest',

                margin=dict(b=20,l=5,r=5,t=40),

                annotations=[ dict(

                    text= "Graph Visualization program",

                    showarrow=False,

                    xref="paper", yref="paper",

                    x=0.005, y=-0.002 ) ],

                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))



#py.iplot(fig, filename='networkx')

#py.offline.iplot(fig, filename='networkx')

plotly.offline.iplot(fig, filename='networkx')