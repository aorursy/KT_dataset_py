import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import networkx as nx
%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')
# Creating graph objects
G = nx.Graph()

#Adding first node
G.add_node(1)

#Adding more nodes
G.add_nodes_from([2,3,4,5,6,8,9,12,15,16])

#Drawing the graph
nx.draw(G)
#Adding edges
G.add_edges_from([(2,4),(2,6),(2,8),(2,12),(2,16),(3,6),(3,9), (3,12),(3,15),(4,8),(4,12),(4,16),(6,12),(8,16)])

#Drawing the graph
nx.draw(G)
#Drawing the graph in circular mode
nx.draw_circular(G)
#Drawing the graph in spring mode
nx.draw_spring(G)
#Labeling and coloring graph nodes
nx.draw_circular(G, node_color='bisque', with_labels=True)
#Removing a node
G.remove_node(1)
nx.draw_circular(G, node_color='bisque', with_labels=True)
#Identifying graph properties
sum_stats = nx.info(G)
print(sum_stats)
#Viewing degree of each node 
print(nx.degree(G))
#Using graph generators
#Creating a complete graph
G = nx.complete_graph(25)
nx.draw(G, node_color='bisque', with_labels=True)
#Cerating a random directed graph with 7 nodes
G = nx.gnc_graph(7, seed=25)
nx.draw(G, node_color='bisque', with_labels=True)
#Using sub graph generator.
ego_G = nx.ego_graph(G, 3, radius=5)
nx.draw(G, node_color='bisque', with_labels=True)
#Simulating a social network (ie; directed network analysis)
#Generating a graph object and edgelist

DG = nx.gn_graph(7, seed=25)

for line in nx.generate_edgelist(DG, data=False): print(line)
#Viewing node attributes (it's currently empty)
print(DG.node[0])
#Assigning attributes to nodes
DG.node[0]['name'] = 'Alice'
print(DG.node[0])
DG.node[1]['name'] = 'Bob'
DG.node[2]['name'] = 'Claire'
DG.node[3]['name'] = 'Dennis'
DG.node[4]['name'] = 'Esther'
DG.node[5]['name'] = 'Frank'
DG.node[6]['name'] = 'George'
#Adding age attribute using add_nodes_from
DG.add_nodes_from([(0,{'age':25}),(1,{'age':31}),(2,{'age':18}),(3,{'age':47}),(4,{'age':22}),
                   (5,{'age':23}),(6,{'age':50})])
#Adding gender
DG.node[0]['gender'] = 'f'
DG.node[1]['gender'] = 'm'
DG.node[2]['gender'] = 'f'
DG.node[3]['gender'] = 'm'
DG.node[4]['gender'] = 'f'
DG.node[5]['gender'] = 'm'
DG.node[6]['gender'] = 'm'
#Viewing node attributes 
print(DG.node[0])
#Visualizing the network graph
nx.draw_circular(DG, node_color='bisque', with_labels=True)
#Labeling the graph nodes using names instead of node numbers
labeldict = {0: 'Alice',1:'Bob',2:'Claire',3:'Dennis',4:'Esther',5:'Frank',6:'George'}

nx.draw_circular(DG, labels=labeldict, node_color='bisque', with_labels=True)
#Transforming the directed graph to undirected
G = DG.to_undirected()
nx.draw_spectral(G, labels=labeldict, node_color='bisque', with_labels=True)
print(nx.info(DG))
#Considering degrees in a social network
#We can use degree in a directed graph to identify the most influential node
#Alice is the most influential in this graph
DG.degree()
#Identifying successor nodes (nodes that can replace original nodes)
nx.draw_circular(DG, node_color='bisque', with_labels=True)
#Let's find node 3's successor
list(DG.successors(3))
#Identifying neighbors (only out connections are considered)
list(DG.neighbors(4))
#Identifying neighbors
list(G.neighbors(4))
