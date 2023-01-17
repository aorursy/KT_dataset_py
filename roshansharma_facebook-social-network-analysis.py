# for some basic operations

import numpy as np 

import pandas as pd 



# for basic visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for network visualizations

import networkx as nx
# Undirected Graphs



g = nx.Graph()

g.add_edge('A', 'B')

g.add_edge('B', 'C')

g.add_edge('C', 'D')

g.add_edge('B', 'D')

g.add_edge('A', 'E')

g.add_edge('A', 'F')

g.add_edge('A', 'G')



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')



pos = nx.spring_layout(g)



# drawing nodes

nx.draw_networkx_nodes(g, pos, node_size = 900, node_color = 'orange')



# drawing edges

nx.draw_networkx_edges(g, pos, width = 6, alpha = 0.5, edge_color = 'black')



# labels

nx.draw_networkx_labels(g, pos, font_size = 20, font_family = 'sans-serif')



plt.title('Undirected Graphs', fontsize = 20)

plt.axis('off')

plt.show()
# Directed Graphs



g = nx.DiGraph()

g.add_edge('A', 'B')

g.add_edge('B', 'C')

g.add_edge('C', 'H')

g.add_edge('B', 'D')

g.add_edge('A', 'E')

g.add_edge('A', 'F')

g.add_edge('A', 'G')



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')



pos = nx.spring_layout(g)



# drawing nodes

nx.draw_networkx_nodes(g, pos, node_size = 900, node_color = 'yellow')



# drawing edges

nx.draw_networkx_edges(g, pos, edge_color = 'brown', width = 6, alpha = 0.5)



# defining labels

nx.draw_networkx_labels(g, pos, font_size=20, font_family='sans-serif')



plt.title('Directed Graphs', fontsize = 20)

plt.axis('off')

plt.show()
# weighted networks



# Undirected Graphs



g = nx.Graph()

g.add_edge('A', 'B', weight = 8)

g.add_edge('B', 'C', weight = 12)

g.add_edge('C', 'J', weight = 15)

g.add_edge('B', 'D', weight = 3)

g.add_edge('A', 'E', weight = 5)

g.add_edge('A', 'F', weight = 18)

g.add_edge('A', 'G', weight = 10)



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')

plt.title('Weighted Networks', fontsize = 20)



elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] <  10]

esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['weight'] >= 10]



pos = nx.spring_layout(g)  



# nodes

nx.draw_networkx_nodes(g, pos, node_size = 900, node_color = 'pink')



# edges

nx.draw_networkx_edges(g, pos, edgelist = elarge, width = 6)

nx.draw_networkx_edges(g, pos, edgelist = esmall, width = 6, alpha = 0.5, edge_color = 'b', style = 'dashed')



# labels

nx.draw_networkx_labels(g, pos, font_size = 20, font_family = 'sans-serif')



plt.axis('off')

plt.show()
# signed networks



# Undirected Graphs



g = nx.Graph()

g.add_edge('A', 'B', sign = '+')

g.add_edge('B', 'C', sign = '-')

g.add_edge('C', 'J', sign = '+')

g.add_edge('B', 'D', sign = '-')

g.add_edge('A', 'E', sign = '+')

g.add_edge('A', 'F', sign = '+')

g.add_edge('A', 'G', sign = '-')



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')

plt.title('Signed Networks', fontsize = 20)



elarge = [(u, v) for (u, v, d) in g.edges(data=True) if d['sign'] == '+']

esmall = [(u, v) for (u, v, d) in g.edges(data=True) if d['sign'] == '-']



pos = nx.spring_layout(g)  



# nodes

nx.draw_networkx_nodes(g, pos, node_size=700)



# edges

nx.draw_networkx_edges(g, pos, edgelist=elarge,

                       width=6)

nx.draw_networkx_edges(g, pos, edgelist=esmall,

                       width=6, alpha=0.5, edge_color='b', style='dashed')



# labels

nx.draw_networkx_labels(g, pos, font_size=20, font_family='sans-serif')



plt.axis('off')

plt.show()
# relation networks



# Undirected Graphs



g = nx.Graph()

g.add_edge('A', 'B', relation = 'family')

g.add_edge('B', 'C', relation = 'friend')

g.add_edge('C', 'J', relation = 'coworker')

g.add_edge('B', 'D', relation = 'family')

g.add_edge('A', 'E', relation = 'friend')

g.add_edge('A', 'F', relation = 'coworker')

g.add_edge('A', 'G', relation = 'friend')



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')

plt.title('Relation based Networks', fontsize = 20)



pos = nx.spring_layout(g)  



# nodes

nx.draw_networkx_nodes(g, pos, node_size = 700, node_color = 'lightgreen')



# edges

nx.draw_networkx_edges(g, pos, width = 6, alpha = 0.5, edge_color = 'black')



# labels

nx.draw_networkx_labels(g, pos, font_size = 20, font_family = 'sans-serif')



plt.axis('off')

plt.show()
# See what layouts are available in networkX



[x for x in nx.__dir__() if x.endswith('_layout')]
# bipartite graphs



from networkx.algorithms import bipartite



B = nx.Graph()



B.add_nodes_from(['A','B','C','D','E'], bipartite = 0)

B.add_nodes_from([1, 2, 3, 4], bipartite = 1)

B.add_edges_from([('A', 1),('B', 1),('C', 1),('C', 3),('D', 2),('E',3),('E',4)])



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')

plt.title('Bi-Partite Networks', fontsize = 20)



pos = nx.shell_layout(B)  



# nodes

nx.draw_networkx_nodes(B, pos, node_size = 700, node_color = 'lightblue')



# edges

nx.draw_networkx_edges(B, pos, width = 6, alpha = 0.5, edge_color = 'black')



# labels

nx.draw_networkx_labels(B, pos, font_size = 20, font_family = 'sans-serif')



plt.axis('off')

plt.show()
# we can also check whether a graph is bipartite or not



bipartite.is_bipartite(B)
# checking if a set of nodes is a bipartition of a graph



X = set([1, 2, 3, 4])

bipartite.is_bipartite_node_set(B, X)
# projected graphs



B = nx.Graph()

B.add_edges_from([('A', 1),('B', 1),('C', 1),('D', 1),('H', 1),('B', 2),('C', 2),('D', 1),

                 ('H', 1),('B', 2),('C', 2),('D', 2),('E', 2),('G', 2),('E', 3),('F', 3),

                 ('H', 3), ('J', 3), ('E', 4), ('I',4), ('J', 4)])



X = set(['A','B','C','D','E','F','G','H','I','J'])

P = bipartite.projected_graph(B, X)



pos = nx.circular_layout(P)



nx.draw_networkx_nodes(P, pos, node_color = 'cyan')



nx.draw_networkx_edges(P, pos, edge_color = 'magenta', width = 6, alpha = 0.5)



nx.draw_networkx_labels(P, pos, font_size = 20, font_family = 'sans-serif')



plt.title('Projected Graph', fontsize = 20)

plt.axis('off')

plt.show()
a = nx.Graph()



a.add_edge('A', 'B')

a.add_edge('A', 'K')

a.add_edge('B', 'C')

a.add_edge('C', 'F')

a.add_edge('F', 'E')

a.add_edge('F', 'G')

a.add_edge('C', 'E')

a.add_edge('E', 'D')

a.add_edge('E', 'H')

a.add_edge('K', 'B')

a.add_edge('E', 'I')

a.add_edge('I', 'J')



a = nx.bfs_tree(a, 'A')

pos = nx.kamada_kawai_layout(a)

nx.draw_networkx(a, size = 900)

plt.axis('off')

plt.title('BFS Tree')

plt.show()
# let's check the edges of the tree



a.edges()
# let's check the shortest path from A



nx.shortest_path_length(a, 'A')
# checking the average shortest path length



nx.average_shortest_path_length(a)
import matplotlib.pyplot as plt

import networkx as nx



G = nx.random_geometric_graph(200, 0.125)

# position is stored as node attribute data for random_geometric_graph

pos = nx.get_node_attributes(G, 'pos')



# find node near center (0.5,0.5)

dmin = 1

ncenter = 0

for n in pos:

    x, y = pos[n]

    d = (x - 0.5)**2 + (y - 0.5)**2

    if d < dmin:

        ncenter = n

        dmin = d



# color by path length from node near center

p = dict(nx.single_source_shortest_path_length(G, ncenter))



plt.rcParams['figure.figsize'] = (10, 10)

nx.draw_networkx_edges(G, pos, nodelist=[ncenter], alpha=0.4)

nx.draw_networkx_nodes(G, pos, nodelist=list(p.keys()),

                       node_size=80,

                       node_color=list(p.values()),

                       cmap=plt.cm.Reds_r)



plt.title('Random Geometric Graph', fontsize = 20)

plt.xlim(-0.05, 1.05)

plt.ylim(-0.05, 1.05)

plt.axis('off')

plt.show()
import matplotlib.pyplot as plt

import networkx as nx



G = nx.karate_club_graph()



plt.style.use('fivethirtyeight')

nx.draw_circular(G, with_labels=True)



plt.title(' Karate Club Networks')

plt.show()
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
# checking the betweenness centrality 



bc = nx.betweenness_centrality(fb)

bc
# checking the degree of each node in the network



degree = nx.degree_histogram(fb)

degree
print(fb.order())

print(fb.size())