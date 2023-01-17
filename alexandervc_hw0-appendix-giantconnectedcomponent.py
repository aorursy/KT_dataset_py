try:
    import igraph # igraph is already preinstalled on kaggle, but not colab  
except:    
    !pip install python-igraph # Pay attention: not just "pip install igraph" 
    !pip install cairocffi # Module required for plots 
    import igraph # igraph is already preinstalled on kaggle, but not colab  

import numpy as np
import matplotlib.pyplot as plt
import time 

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree


import igraph
#n = 5242;  m = 14484
#n = 5 ; k = 2  
def create_toy_knn_graph(n,k):
  g = igraph.Graph( directed=True)#  G = snap.TUNGraph.New(n,m) # Allocate memory for UNdirected graph n-nodes, m-edges
  g.add_vertices(n)
  
  for i in range(n): # Add m edges connected at random 
    list_target_nodes = []
    while True: # technical loop - check we are not adding already existing edges 
      v2 = np.random.randint(0,n )
      if i == v2:         continue
      if v2 in list_target_nodes:   continue
      list_target_nodes.append(v2)
      g.add_edge(i,v2)
      if len(list_target_nodes) >= k: 
        break
  return g

n = 45 ; k = 1 
g =  create_toy_knn_graph(n,k)
h = g.degree_distribution(bin_width=1, mode = "out", )
#print(h)

print(" Random graph is created by the rule - each node has one random out-going edge to other node. As you can see from the plots typically the graph falls into one connected component or one big and 1-2 small" )

r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
list_components_sizes = [ len(t) for t in r  ]
print("Sizes of connected components")
print(list_components_sizes)
            
igraph.plot(g, bbox = (600,500))

k = 1
for n in [1e1, 1e3,1e4,1e5]:
    n = int(n)
    t0 = time.time()
    g =  create_toy_knn_graph(n,k)
    r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
    print("n_nodes = ", n, "seconds passed", time.time()-t0,  "Connected component sizes:" )
    list_components_sizes = [ len(t) for t in r  ]
    print(list_components_sizes)
    print()
dim = 100
        
t0 = time.time()    
c = 0        
for n in [1e4]:
    n = int(n)    
    X = np.random.randn(n, dim)
    print("Dimension",dim," n ", n)
    nbrs = NearestNeighbors(n_neighbors=2  ).fit(X) # 'ball_tree'
    distances, indices = nbrs.kneighbors(X)
    g = igraph.Graph( directed = True )
    g.add_vertices(range(n))
    g.add_edges(indices )
    r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
    list_components_sizes = [ len(t) for t in r  ]
    print("10 largest components sizes")
    print(np.sort(list_components_sizes)[::-1][:10])    
    print("Maximum size of the connected component")
    print(np.max(list_components_sizes)    )
    print(time.time() - t0 , "seconds passed" )
    plt.hist(list_components_sizes)
    plt.title("Histogram of components sizes")
    plt.show()
dim = 2
        
t0 = time.time()    
c = 0        
for n in [1e2]:
    n = int(n)    
    X = np.random.randn(n, dim)
    print("Dimension",dim," n ", n)
    nbrs = NearestNeighbors(n_neighbors=2  ).fit(X) # 'ball_tree'
    distances, indices = nbrs.kneighbors(X)
    g = igraph.Graph( directed = True )
    g.add_vertices(range(n))
    g.add_edges(indices )
    r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
    list_components_sizes = [ len(t) for t in r  ]
    print("10 largest components sizes")
    print(np.sort(list_components_sizes)[::-1][:10])    
    print("Maximum size of the connected component")
    print(np.max(list_components_sizes)    )
    print(time.time() - t0 , "seconds passed" )
    plt.hist(list_components_sizes)
    plt.title("Histogram of components sizes")
    plt.show()
    
    
igraph.plot(g)    
