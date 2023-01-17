def configure_plotly_browser_state():

  import IPython

  display(IPython.core.display.HTML('''

        <script src="/static/components/requirejs/require.js"></script>

        <script>

          requirejs.config({

            paths: {

              base: '/static/base',

              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',

            },

          });

        </script>

        '''))
import matplotlib.pyplot as plt

from networkx import nx

from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot

init_notebook_mode(connected=True)



n = 50  # 50 nodes

p = 0.2  # probability of edge between nodes



G = nx.erdos_renyi_graph(n,p) # sample graph
nx.draw(G)

plt.show()
G.edges()
G.nodes()
import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline
G_fb = nx.read_edgelist("../input/facebook_combined.txt", create_using = nx.Graph(), nodetype = int)
#Quick snapshot of the Network

print (nx.info(G_fb))
#Create network layout for visualizations

spring_pos = nx.spring_layout(G_fb)
plt.axis("off") # using normal networkx

nx.draw_networkx(G_fb, pos = spring_pos, with_labels = False, node_size = 35)
pos = nx.spring_layout(G_fb)

betCent = nx.betweenness_centrality(G_fb, normalized=True, endpoints=True)

node_color = [20000.0 * G_fb.degree(v) for v in G_fb]

node_size =  [v * 10000 for v in betCent.values()]

plt.figure(figsize=(20,20))

nx.draw_networkx(G_fb, pos=pos, with_labels=False,

                 node_color=node_color,

                 node_size=node_size )

plt.axis('off')
import plotly

def with_plotly(G,title):

  

  labels=list(G.nodes()) # labels are the node names

  pos=nx.fruchterman_reingold_layout(G)  

   

  Xn=[pos[k][0] for k in range(len(pos))]

  Yn=[pos[k][1] for k in range(len(pos))]

  

  trace_nodes=dict(type='scatter',x=Xn,y=Yn,mode='markers',marker=dict(size=28, color='rgb(0,240,0)'),text=labels,hoverinfo='text')

 

  Xe=[]

  Ye=[]

  for e in G.edges():

      Xe.extend([pos[e[0]][0], pos[e[1]][0], None])

      Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

  

  trace_edges=dict(type='scatter',mode='lines',x=Xe,y=Ye,line=dict(width=1, color='rgb(25,25,25)'),hoverinfo='none')

  

  axis=dict(showline=False,zeroline=False,showgrid=False,showticklabels=False,title='')

  layout=dict(title= title,font= dict(family='Balto'),autosize=True,showlegend=False,xaxis=axis,yaxis=axis,margin=dict(l=40,r=40,b=85,t=100,pad=0,),hovermode='closest',plot_bgcolor='#efecea')



  fig = dict(data=[trace_edges, trace_nodes], layout=layout)

  plotly.offline.iplot(fig)  
configure_plotly_browser_state()

title='FB Graph'

with_plotly(G_fb,title)
# Define get_nodes_and_nbrs()

def get_nodes_and_nbrs(G, nodes_of_interest):

    """

    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.

    """

    nodes_to_draw = []



    # Iterate over the nodes of interest

    for n in nodes_of_interest:



        # Append the nodes of interest to nodes_to_draw

        nodes_to_draw.append(n)



        # Iterate over all the neighbors of node n

        for nbr in G.neighbors(n):



            # Append the neighbors of n to nodes_to_draw

            nodes_to_draw.append(nbr)



    return G.subgraph(nodes_to_draw)



# Extract the subgraph with the nodes of interest: T_draw

nodes_of_interest = [2,5]  # can be changed

G_draw = get_nodes_and_nbrs(G_fb, nodes_of_interest)



# Draw the subgraph to the screen

nx.draw(G_draw,with_labels=True)

plt.show()
labels=list(G_draw.nodes()) # labels are the node names

pos=nx.spring_layout(G_draw)  

Xn=[pos[k][0] for k in pos]

Yn=[pos[k][1] for k in pos]



trace_nodes=dict(type='scatter',x=Xn,y=Yn,mode='markers',marker=dict(size=28,color=[],colorbar=dict(thickness=15,title='Node Connections',

                 xanchor='left', titleside='right')),text=[],hoverinfo='text')



Xe=[]

Ye=[]

for e in G_draw.edges():

    Xe.extend([pos[e[0]][0], pos[e[1]][0], None])

    Ye.extend([pos[e[0]][1], pos[e[1]][1], None])



trace_edges=dict(type='scatter',mode='lines',x=Xe,y=Ye,line=dict(width=1),hoverinfo='none')



axis=dict(showline=False,zeroline=False,showgrid=False,showticklabels=False,title='')

layout=dict(title= 'FB Subgraph',font= dict(family='Balto'),autosize=True,showlegend=False,xaxis=axis,yaxis=axis,margin=dict(l=40,r=40,b=85,t=100,pad=0,),hovermode='closest',plot_bgcolor='#efecea')

i=0

for node, adjacencies in enumerate(G_draw.adjacency()):

    trace_nodes['marker']['color']+=tuple([len(adjacencies[1])])

    node_info = 'Node:'+str(labels[i])+'\n | # of connections: '+str(len(adjacencies[1]))

    trace_nodes['text']+=tuple([node_info])

    i+=1

    



configure_plotly_browser_state()

fig = dict(data=[trace_edges, trace_nodes], layout=layout)

plotly.offline.iplot(fig) 
import community

import matplotlib.pyplot as plt

# example community detection

G = nx.karate_club_graph()

part = community.best_partition(G)

values = [part.get(node) for node in G.nodes()]



nx.draw_spring(G, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=False)

G=G_draw # Community Detection for the nodes of interest

part = community.best_partition(G)

values = [part.get(node) for node in G.nodes()]



nx.draw_spring(G, cmap = plt.get_cmap('Pastel1'), node_color = values, node_size=450, with_labels=True)
