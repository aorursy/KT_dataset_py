# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools

flatten = itertools.chain.from_iterable

from collections import defaultdict

import networkx as nx

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class Graph():

    def __init__(self):

        self.edges = defaultdict(list)

        self.weights = {}



    def printArr(self, dist):

        print("Vertex Distance from Source") 

        for i in range(self.V):

            print("% d \t\t % d" % (i, dist[i])) 

    

    def add_edge(self, from_node, to_node, weight):

        # Note: assumes edges are bi-directional

        self.edges[from_node].append(to_node)

        self.edges[to_node].append(from_node)

        self.weights[(from_node, to_node)] = weight

        self.weights[(to_node, from_node)] = weight

        



    def get_all_vertices(self):

        all_vertices = set(list(flatten([[x[0], *x[1]] for x in list(graph.edges.items())])))

        return all_vertices



    def get_all_edges_with_weight(self):

        all_edges = list()

        for v, list_u in graph.edges.items():

            for u in list_u:

                edge =  [*(v, u), graph.weights[(v, u)]]

                all_edges.append(edge)

        return all_edges

    



    

def dijsktra(graph, initial):

    dists = dict()

    paths = dict()

    all_vertices = graph.get_all_vertices()

    

    for end in all_vertices:

        shortest_paths = {initial: (None, 0)}

        current_node = initial

        visited = set()



        while current_node != end:

            visited.add(current_node)

            destinations = graph.edges[current_node]

            weight_to_current_node = shortest_paths[current_node][1]



            for next_node in destinations:

                weight = graph.weights[(current_node, next_node)] + weight_to_current_node

                if next_node not in shortest_paths:

                    shortest_paths[next_node] = (current_node, weight)

                else:

                    current_shortest_weight = shortest_paths[next_node][1]

                    if current_shortest_weight > weight:

                        shortest_paths[next_node] = (current_node, weight)



            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}

            if not next_destinations:

                return "Route Not Possible"



            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])



        # Work back through destinations in shortest path

        path = []

        distance = 0

        while current_node is not None:

            path.append(current_node)

            next_node = shortest_paths[current_node][0]

            current_node = next_node

        for i in range(len(path)-1):

            distance += graph.weights[(path[i], path[i+1])] 



        # Reverse path

        path = path[::-1]

        dists[end]=distance

        paths[end] = path

        

    return dists, paths



def get_path(prev, v):

    path  = list()

    path.append(v)

    while prev[v] != None:

        v = prev[v]

        path.append(v)

    

    path.reverse()

    return path



def BellmanFord(graph, src):  

    all_vertices = graph.get_all_vertices()

    num_vertices = len(all_vertices)



    dist =  {v: np.inf for v in all_vertices} 

    prev = {v: None for v in all_vertices}  

    dist[src] = 0



    all_edges = graph.get_all_edges_with_weight()



    for i in range(num_vertices-1):  

        for u, v, w in all_edges:  

            if dist[v] > dist[u] + w:  

                dist[v] = dist[u] + w  

                prev[v] = u





    for u, v, w in all_edges:  

            if dist[v] >  dist[u] + w:  

                print("Negative weight cycle!") 

                return None



    return dist, prev

graph = Graph()



edges = [

    ('CSQ', 'LSC', 200),

    ('CSQ', 'PC', 300),

    ('PC', 'PD', 100),

    ('PC', 'CS', 80),

    ('PC', 'TL', 30),

    ('PD', 'OM', 200),

    ('PD', 'FA', 50),

    ('PD', 'SHC', 100),

    ('SHC', 'SC', 50),

    ('SHC', 'BH', 200),

    ('LSC', 'SLH', 250),

    ('LSC', 'CS', 150),

    ('CS', 'B', 30),

    ('CS', 'TL', 40),

    ('TL', 'B', 80),

    ('TL', 'OM', 30),

    ('OM', 'MH', 100),

    ('OM', 'FA', 90),

    ('FA', 'SC', 80),

    ('FA', 'MH', 180),

    ('SC', 'MH', 100),

    ('SC', 'W', 100),

    ('SC', 'NBB', 110),

    ('B', 'SLH', 100),

    ('B', 'MH', 200),

    ('B', 'MC', 300),

    ('MH', 'W', 50),

    ('MH', 'MC', 150),

    ('W', 'NBB', 50),

    ('W', 'MC', 100),

    ('NBB', 'MC', 150),

    ('NBB', 'OTA', 30),

    ('NBB', 'BH', 20),

    ('BH', 'BVA', 350),

    ('BH', 'OTA', 40),

    ('SLH', 'MC', 120),

    ('MC', 'OTA', 160)

    ]



for edge in edges:

    graph.add_edge(*edge)



all_vertices = set(flatten([ [e[0], e[1]] for e in edges]))

all_vertices.remove("CS")
#

pos = {'B':[50,-40],

 'BH':[50,-160],

 'BVA':[50,-180],

 'CSQ':[0,0],

 'FA':[30,-100],

 'LSC':[20,-20],

 'MC':[70,-70],

 'MH':[50,-100],

 'NBB':[50,-140],

 'OM':[30,-80],

 'OTA':[70,-140],

 'PC':[0,-60],

 'PD':[-10,-100],

 'SC':[30,-120],

 'SHC':[-10,-140],

 'SLH':[70,-20],

 'TL':[30,-60],

 'W':[50,-120],

 'CS':[30,-40]}

# draw function

def drawNetwork(edgelist,pos):

    # Build a dataframe with your connections

    # This time a pair can appear 2 times, in one side or in the other!

    df = pd.DataFrame({ 'from':[e[0] for e in edgelist], 'to':[ e[1] for e in edgelist],'weight':[ e[2] for e in edgelist]})

    df



    # Build your graph. Note that we use the DiGraph function to create the graph!

    G=nx.from_pandas_edgelist(df, 'from', 'to',['weight'] )



    plt.figure(figsize=(5,7))

    # Make the graph

    nx.draw(G, with_labels=True, node_size=800, alpha=1, arrows=False, edge_attr=True, pos=pos)

    # Draw the edge labels

    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G,'weight'))

    # Remove the axis

    plt.axis('off')

    # Show the plot

    return G



# using nx lib to draw and check correct shortest path

G_nx = drawNetwork(edges,pos)
dijsktra_dists, dijsktra_prevs = dijsktra(graph, "CS")
bellmanFord_dists, bellmanFord_prevs = BellmanFord(graph, "CS")



print("Shortest Path from CS")

for v in all_vertices:

    print(f"-------------------------> to {v}")

    ## dijsktra

    dijsktra_d, dijsktra_path = dijsktra_dists[v], dijsktra_prevs[v]

    print(f"using Dijkstra’s Algo is \t {dijsktra_d}, via the path \t {dijsktra_path}")

    

    ## Belman

    Bellman_d, Bellman_path = bellmanFord_dists[v], get_path(bellmanFord_prevs, v) 

    print(f"using Bellman’s Algo is \t {Bellman_d}, via the path \t {Bellman_path}")

    

    ## nx libary

    nx_d =  nx.shortest_path_length(G_nx,source = 'CS', target = v,weight ='weight')

    nx_path =  nx.shortest_path(G_nx,source = 'CS', target = v,weight ='weight')

    print(f"using networkx is \t\t {nx_d}, via the path \t {nx_path}", end="\n")

    

    assert (dijsktra_path == Bellman_path) and (dijsktra_d == Bellman_d), "The results are different!"