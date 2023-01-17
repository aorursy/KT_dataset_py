# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# To install:

# on kaggle - it is already preinstalled 

# !pip install python-igraph # Pay attention: not just "pip install igraph" 

# !pip install cairocffi # Module required for plots 





import igraph # On kaggle it is pre-installed 



import numpy as np

#import pandas as pd

#import os

#import matplotlib.pyplot as plt
g = igraph.Graph()

g.add_vertices(5)

g.add_edge(0, 4) # Connect nodes 0 and 4

g.add_edges([(0,1), (1,2), (3,2) ]) # Add list of edges 

print(g) # print info on graph  

# IGRAPH U--- 5 4 -- # Means "U" - undirected, 5 - n_nodes, 4 - n_edges 
# Nodes and Edges number



g.vcount() , g.ecount()
# Edge list - list of tuples of nodes like (v1,v2) for each connected v1 and v2  

g.get_edgelist()
print( type( g.get_adjacency() ) )

print( g.get_adjacency() )
igraph.plot(g,bbox = (200,100) )
g = igraph.Graph(directed = True)

g.add_vertices(5)

g.add_edge(0, 4) # Connect nodes 0 and 4

g.add_edges([(0,1), (1,2), (3,2) ])



layout = g.layout('kk') # kk - Kamada-Kawai layout - one of the most popular layouts 

# List of possible layouts is here: https://igraph.org/python/doc/igraph.Graph-class.html#layout

visual_style = {}

visual_style["vertex_color"] = ['green' for v in g.vs]

visual_style["vertex_label"] = range(g.vcount()) 

visual_style["vertex_size"] = 30

igraph.plot(g, layout = layout, **visual_style, bbox = (300,100) )
g = igraph.Graph.Tree(40,3) # Lattice([10,10],  circular = False)

#igraph.plot(g,  bbox=(200,200) )

layout = g.layout_grid( ) # reingold_tilford(root=[2])

visual_style = {}

visual_style["vertex_color"] = ['pink' for v in g.vs]

visual_style["vertex_label"] = range(g.vcount()) 

visual_style["vertex_size"] = 20

igraph.plot(g,  **visual_style, bbox = (200,200) ) # layout = layout,
# Here is more advanced example - set layout by hands, shrink plot from margins by 80, put labels on edges, change size of scripts

g2 = igraph.Graph()

g2.add_vertices(3)

g2.add_edges([[0,1],[1,2],[2,0], [2,2] ])

g2.es['label'] = ['A','B','C','D'] # It is misleading to put so much unnecessary info on that graph 



#layout = g2.layout_circle()

layout = igraph.Layout([ (2, 0) , (3, 2), (0, 0), ])

visual_style = {}

visual_style["vertex_color"] = ['red','gray','green']

visual_style["vertex_label"] = ['1','2','3']

visual_style["edge_label_size"] = 40 # [2,2,2]

visual_style["margin"] = 80

visual_style["vertex_size"] = 30



igraph.plot(g2, layout = layout, **visual_style,  bbox = (600,300))
n = 10

m = 20

g = igraph.Graph.Erdos_Renyi(n,  m=m, directed=False, loops=False) # Generates a graph based on the Erdos-Renyi model.

# https://igraph.org/python/doc/igraph.Graph-class.html



igraph.plot(g,  bbox = (300,100))

g = igraph.Graph.Famous("zachary") 

igraph.plot(g, bbox = (500,200))

g = igraph.Graph(directed=True)

g.add_vertices(4)

g.add_edge(0,1)

g.add_edge(1,2)

g.neighbors(0), g.neighbors(1), g.neighbors(2) , g.neighbors(3) 
g = igraph.Graph(directed=True)

g.add_vertices(3)

g.add_edge(0,1)

print( g.degree(), g.indegree(), g.outdegree() )





g = igraph.Graph(directed=True)

g.add_vertices(4)

g.add_edge(0,1)

g.add_edge(0,1) # this creates multi-graph, thus degrees would be doubled

print( g.degree(), g.indegree(), g.outdegree() )

g = igraph.Graph.Famous("zachary") 

h = g.degree_distribution(bin_width=1) # Returns igraph.Histogram object https://igraph.org/python/doc/igraph.statistics.Histogram-class.html

print(h)

print()

print(list(h.bins()) )



g = igraph.Graph(directed=True)

g.add_vertices(4)

g.add_edge(0,1)

for v in g.vs:

    print(v)

print('Edges')

for e in g.es:

    print(e)

    

    
g = igraph.Graph(directed=True)

g.add_vertices(2)

g.add_edge(0,1)

print('Number of strongly connected compoenents', len( g.clusters(mode='STRONG')), 'what are them:', list( g.clusters(mode='STRONG') ) )

print('Number of weakly connected compoenents', len( g.clusters(mode='WEAK')), 'what are them:', list( g.clusters(mode='WEAK') ) )

visual_style = {}

visual_style["vertex_label"] = range(g.vcount()) 

igraph.plot(g, **visual_style,  bbox = (100,40))

########################################################################

# Create test graph - see plot below

########################################################################



g = igraph.Graph()

g.add_vertices(16)

nodes = np.array([0,1,4,5])

for k in [0,2,8,10]:#,2,4,6,8]:

  for i in nodes+k:

    for j in nodes+k:

      if i<=j: continue 

      g.add_edge(i, j)

g.add_edge(1, 2)

g.add_edge(4, 8)

g.add_edge(13, 14)

g.add_edge(7, 11)



########################################################################

# Cluster by Louvain algorithm 

# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel

########################################################################

louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)

modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])

print("The modularity for igraph-Louvain partition is {}".format(modularity1))

#print();

print('Partition info:')

print(louvain_partition)



########################################################################

# Cluster by optimal algorithm (applicable only for small graphs <100 nodes), it would be very slow otherwise 

# https://igraph.org/python/doc/igraph.Graph-class.html#community_optimal_modularity

########################################################################

print();

v = g.community_optimal_modularity() # weights= gra.es["weight"]) 

modularity1 = g.modularity(v)#, weights=graph.es['weight'])

print("The modularity for igraph-optimal partition is {}".format(modularity1))

#print();

print('Partition info:')

print(v) 



########################################################################

# Plot graph 

########################################################################

layout = g.layout_grid( ) # reingold_tilford(root=[2])

visual_style = {} 

dict_colors = {0:'Aqua', 1:'Aqua', 4:'Aqua', 5:'Aqua',2:'Aquamarine', 3:'Aquamarine', 6:'Aquamarine', 7:'Aquamarine',

               8:'Crimson', 9:'Crimson', 12:'Crimson', 13:'Crimson',10:'Goldenrod', 11:'Goldenrod', 14:'Goldenrod', 15:'Goldenrod',

               } # https://en.wikipedia.org/wiki/X11_color_names - colors by names supported by igraph 

visual_style["vertex_color"] = [dict_colors[k]  for k in range(g.vcount() )]

visual_style["vertex_label"] = range(g.vcount()) 

igraph.plot(g, layout = layout, **visual_style, bbox = (200,200) )
matr_A = np.array( [[0,1,0],[0,0,0],[0,1,0]])

g = igraph.Graph().Adjacency(matr_A.tolist())

#g.to_undirected(mode = 'collapse')

igraph.plot(g, bbox = (200,200))
g.to_undirected(mode = 'collapse')

igraph.plot(g, bbox = (200,200))