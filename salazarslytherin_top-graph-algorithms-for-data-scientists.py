# for some basic operations

import numpy as np 

import pandas as pd 



# for basic visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for network visualizations

import networkx as nx
edgelist = [['Mannheim', 'Frankfurt', 85], ['Mannheim', 'Karlsruhe', 80], ['Erfurt', 'Wurzburg', 186], ['Munchen', 'Numberg', 167], ['Munchen', 'Augsburg', 84], 

            ['Munchen', 'Kassel', 502], ['Numberg', 'Stuttgart', 183], ['Numberg', 'Wurzburg', 103], ['Numberg', 'Munchen', 167], ['Stuttgart', 'Numberg', 183], 

            ['Augsburg', 'Munchen', 84], ['Augsburg', 'Karlsruhe', 250], ['Kassel', 'Munchen', 502], ['Kassel', 'Frankfurt', 173], ['Frankfurt', 'Mannheim', 85], 

            ['Frankfurt', 'Wurzburg', 217], ['Frankfurt', 'Kassel', 173], ['Wurzburg', 'Numberg', 103], ['Wurzburg', 'Erfurt', 186], ['Wurzburg', 'Frankfurt', 217], 

            ['Karlsruhe', 'Mannheim', 80], ['Karlsruhe', 'Augsburg', 250],["Mumbai", "Delhi",400],["Delhi", "Kolkata",500],

            ["Kolkata", "Bangalore",600],["TX", "NY",1200],["ALB", "NY",800]]

# Undirected Graphs

'''

['bipartite_layout',

 'circular_layout',

 'kamada_kawai_layout',

 'random_layout',

 'rescale_layout',

 'shell_layout',

 'spring_layout',

 'spectral_layout',

 'fruchterman_reingold_layout']

 '''

g = nx.Graph()

for edge in edgelist:

    g.add_edge(edge[0],edge[1], weight = edge[2])



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (11, 11)

#plt.style.use('fivethirtyeight')



pos = nx.spring_layout(g)



# drawing nodes

nx.draw_networkx(g,pos)

#plt.title('Undirected Graphs', fontsize = 20)

plt.axis('off')

plt.show()
for i, x in enumerate(nx.connected_components(g)):

    print("cc"+str(i)+":",x)
print(nx.shortest_path(g, 'Stuttgart','Frankfurt',weight='weight'))

print(nx.shortest_path_length(g, 'Stuttgart','Frankfurt',weight='weight'))

for x in nx.all_pairs_dijkstra_path(g,weight='weight'):

    print(x)
nx.draw_networkx(nx.minimum_spanning_tree(g))
import os

print(os.listdir('../input/'))
# reading the dataset



fb = nx.read_edgelist('../input/facebook-combined.txt', create_using = nx.Graph(), nodetype = int)
print(nx.info(fb))
pos = nx.spring_layout(fb)



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (20, 15)

plt.axis('off')

nx.draw_networkx(fb, pos, with_labels = False, node_size = 35)

plt.show()
pagerank = nx.pagerank(fb)
import operator

sorted_pagerank = sorted(pagerank.items(), key=operator.itemgetter(1),reverse=True)

print(sorted_pagerank[:5])
first_degree_connected_nodes = list(fb.neighbors(3437))

second_degree_connected_nodes = []

for x in first_degree_connected_nodes:

    second_degree_connected_nodes+=list(fb.neighbors(x))

second_degree_connected_nodes.remove(3437)

second_degree_connected_nodes = list(set(second_degree_connected_nodes))



subgraph_3437 = nx.subgraph(fb,first_degree_connected_nodes+second_degree_connected_nodes,)

pos = nx.spring_layout(subgraph_3437)
import warnings

warnings.filterwarnings('ignore')

node_color = ['yellow' if v == 3437 else 'red' for v in subgraph_3437]

node_size =  [1000 if v == 3437 else 35 for v in subgraph_3437]

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (20, 15)

plt.axis('off')



nx.draw_networkx(subgraph_3437, pos, with_labels = False, node_color=node_color,node_size=node_size )

plt.show()
pos = nx.spring_layout(subgraph_3437)

betweennessCentrality = nx.betweenness_centrality(subgraph_3437,normalized=True, endpoints=True)
node_size =  [v * 10000 for v in betweennessCentrality.values()]

plt.figure(figsize=(20,20))

nx.draw_networkx(subgraph_3437, pos=pos, with_labels=False,

                 node_size=node_size )

plt.axis('off')