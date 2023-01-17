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
!python -m pip install snap-stanford

import snap



import numpy as np



def erdos_renyi(n,m):

  '''

  Generare Erdos-Renyi graph in snap format of Graph

  Input:

  n - number of nodes

  m - number of edges

  '''

  #n = 5242;  m = 14484

  G = snap.TUNGraph.New(n,m) # Allocate memory for UNdirected graph n-nodes, m-edges

  for i in range(n): # Add n-nodes  

    G.AddNode(i)

  for i in range(m): # Add m edges connected at random 

    while True: # technical loop - check we are not adding already existing edges 

      v1 = np.random.randint(0,n )

      v2 = np.random.randint(0,n )

      if v1 == v2: 

        continue

      if G.IsEdge(v1,v2):

        continue

      G.AddEdge(v1,v2)

      break

  return G



G = erdos_renyi(5242,14484)

print(G.GetNodes(), G.GetEdges() )    

print('Mean Degree', np.mean( [n.GetDeg() for n in G.Nodes()] ),

      'Median Degree', np.median( [n.GetDeg() for n in G.Nodes()] ),

      'Max Degree', np.max( [n.GetDeg() for n in G.Nodes()] ),

      'Min Degree', np.min( [n.GetDeg() for n in G.Nodes()] ) )

import numpy as np



def watts_strogatz (n, n_pairs_additionally_connected  ):

  '''

  Generate small world (Watts-Strogatz) model, i.e. first generate circle graph, than make additional connections i,i+2 nodes (round-upped),

  and then additionally connect several radomly selected nodes (not previously connected). 



  Ouput: G - snap undirected graph ( snap.TUNGraph.New() )

  '''

  #n = 5242

  #n_pairs_additionally_connected  = 4000 



  G = snap.TUNGraph.New() # UNdirected graph 

  for i in range(n): # Add n-nodes  

    G.AddNode(i)



  G.AddEdge(0, n-1) # Round-up edge to make circle 

  for i in range(n-1): # connect i,i+1 nodes

    G.AddEdge(i, i+1) # C

  G.AddEdge(0, n-2) #  

  G.AddEdge(1, n-1) #  

  for i in range(n-2): # connect i,i+2 nodes

    G.AddEdge(i, i+2) # C



  for i in range(n_pairs_additionally_connected): # Connect n_pairs_additionally_connected  Add m edges connected at random 

    while True: # technical loop - check we are not adding already existing edges 

      v1 = np.random.randint(0,n )

      v2 = np.random.randint(0,n )

      if v1 == v2: 

        continue

      if G.IsEdge(v1,v2):

        continue

      G.AddEdge(v1,v2)

      break



  return G



G =  watts_strogatz (5242, 4000  )

print(G.GetNodes(), G.GetEdges() ) 

print('Mean Degree', np.mean( [n.GetDeg() for n in G.Nodes()] ),

      'Median Degree', np.median( [n.GetDeg() for n in G.Nodes()] ),

      'Max Degree', np.max( [n.GetDeg() for n in G.Nodes()] ),

      'Min Degree', np.min( [n.GetDeg() for n in G.Nodes()] ) )





def load_CoathorsArxivGrQc():

    filepath = "/kaggle/input/ml-in-graphs-hw1/ca-GrQc.txt"

    import os

    print('File exists: ',  os.path.isfile(filepath  )  )



    G = snap.LoadEdgeList(snap.PUNGraph,filepath,0,1) 

    snap.DelSelfEdges(G)

    return G

G = load_CoathorsArxivGrQc()

print(G.GetNodes(), G.GetEdges() ) 

print('Mean Degree', np.mean( [n.GetDeg() for n in G.Nodes()] ),

      'Median Degree', np.median( [n.GetDeg() for n in G.Nodes()] ),

      'Max Degree', np.max( [n.GetDeg() for n in G.Nodes()] ),

      'Min Degree', np.min( [n.GetDeg() for n in G.Nodes()] ) )
import matplotlib.pyplot as plt

import numpy as np



plt.style.use('ggplot')

fig = plt.figure( figsize = (15,6 ) )

plt.suptitle('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')



for i,graph_type in enumerate(['ErdosRenyi','SmallWorld','CoathorsArxivGrQc']):

    if graph_type == 'ErdosRenyi':

        G = erdos_renyi(5242,14484)

        if 0: # Built-in way for Erdos-Renyi:

            G = snap.GenRndGnm(snap.PNGraph, 5242,14484) # 1000, 10000)

    if graph_type == 'SmallWorld':

        G =  watts_strogatz(5242, 4000  )

    if graph_type == 'CoathorsArxivGrQc':

        G = load_CoathorsArxivGrQc()



    CntV = snap.TIntPrV()

    snap.GetOutDegCnt(G, CntV)

    list_degs = []

    list_counts = []

    for p in CntV:

      #print("degree %d: count %d" % (p.GetVal1(), p.GetVal2()))

      list_degs.append(p.GetVal1() )

      list_counts.append(p.GetVal2() )



    fig.add_subplot(1,2,1)

    plt.loglog(list_degs, list_counts, linestyle = 'dotted',  label = graph_type)# color = 'b', #Collaboration Network')

    plt.xlabel('Node Degree (log)')

    plt.ylabel('Proportion of Nodes with a Given Degree (log)')

    plt.title('loglog plot')

    plt.legend()



    fig.add_subplot(1,2,2)

    plt.plot(list_degs, list_counts, linestyle = 'dotted', label = graph_type)# color = 'b',  #Collaboration Network')

    plt.xlabel('Node Degree')

    plt.ylabel('Proportion of Nodes with a Given Degree')

    plt.title('NOT loglog plot')

    plt.legend()

plt.show()



def clustering_coef(G , mode_for_end_nodes = 'put_zeros' ):

  '''

  Calculate vector of clustering coefficient for each node 

  input G - snap undirected Graph 

  mode_for_end_nodes = 'nan' - end/disconnected nodes will be ignored i.e. non included in output list

  mode_for_end_nodes = 'put_zero' - assign zero for end/disconnected



  Example:

  # Create triangle graph 

  Gsmall = snap.TUNGraph.New() 

  Gsmall.AddNode(1) # Add nodes 

  Gsmall.AddNode(2)

  Gsmall.AddNode(3)

  Gsmall.AddEdge(1,2)

  Gsmall.AddEdge(1,3)

  Gsmall.AddEdge(2,3)

 

  list_clusterning_coefs4nodes = clustering_coef(Gsmall)

  # We get [1,1,1] - all clustering coefs = 1

  '''

  list_clusterning_coefs4nodes = []

  for n in G.Nodes():

    NodeVec = snap.TIntV()

    snap.GetNodesAtHop(G, n.GetId(), 1, NodeVec, False) # Get neigbours of current node # https://snap.stanford.edu/snappy/doc/reference/GetNodesAtHop.html

    current_degree = len(NodeVec) # same as n.GetDeg()

    if current_degree <= 1: # skip disconnected&end nodes - impossible to calculate for them - getting division by zero

      if mode_for_end_nodes == 'nan':

        continue 

      else:

        list_clusterning_coefs4nodes.append(0)

        continue

    count_edges_between_neigbours = 0

    for neigbor1 in NodeVec:

      for neigbor2 in NodeVec:

        if neigbor1 >= neigbor2:

          continue

        if G.IsEdge(neigbor1, neigbor2):

          count_edges_between_neigbours += 1

    clustering_coef4current_node = 2*count_edges_between_neigbours/ (current_degree * (current_degree-1)  )

    list_clusterning_coefs4nodes.append(clustering_coef4current_node)

  return list_clusterning_coefs4nodes







Gsmall = snap.TUNGraph.New() 

Gsmall.AddNode(1) # Add nodes 

Gsmall.AddNode(2)

Gsmall.AddNode(3)

Gsmall.AddEdge(1,2)

Gsmall.AddEdge(1,3)

Gsmall.AddEdge(2,3)



list_clusterning_coefs4nodes = clustering_coef(Gsmall)

list_clusterning_coefs4nodes
import numpy

for i,graph_type in enumerate(['ErdosRenyi','SmallWorld','CoathorsArxivGrQc']):

    if graph_type == 'ErdosRenyi':

        G = erdos_renyi(5242,14484)

        if 0: # Built-in way for Erdos-Renyi:

            G = snap.GenRndGnm(snap.PNGraph, 5242,14484) # 1000, 10000)

    if graph_type == 'SmallWorld':

        G =  watts_strogatz(5242, 4000  )

    if graph_type == 'CoathorsArxivGrQc':

        G = load_CoathorsArxivGrQc()

        

    list_clusterning_coefs4nodes = clustering_coef(G , mode_for_end_nodes = 'put_zeros' )



    # Compare with built-in implementation:

    GraphClustCoeff = snap.GetClustCf (G, -1)



    print(graph_type,'Clustering coef:', np.mean(list_clusterning_coefs4nodes), 

          'same by built-in function:', GraphClustCoeff)

### Teting Clustering coef. Triangle graph example - get [1,1,1] - correct



print('Triangle graph')

Gsmall = snap.TUNGraph.New() 

Gsmall.AddNode(1) # Add nodes 

Gsmall.AddNode(2)

Gsmall.AddNode(3)

Gsmall.AddEdge(1,2)

Gsmall.AddEdge(1,3)

Gsmall.AddEdge(2,3)



list_clusterning_coefs4nodes = clustering_coef(Gsmall)

print(list_clusterning_coefs4nodes )



print('Check with built-in function:')

NIdCCfH = snap.TIntFltH()

snap.GetNodeClustCf(Gsmall, NIdCCfH)

for item in NIdCCfH:

    print(item, NIdCCfH[item])

print()

print()





### Testing  Clustering coef. Square graph example - get [0,0,0,0] - correct

print('Square graph')

G = snap.TUNGraph.New() 

G.AddNode(1) # Add nodes 

G.AddNode(2)

G.AddNode(3)

G.AddNode(4)

G.AddEdge(1,2)

G.AddEdge(2,3)

G.AddEdge(3,4)

G.AddEdge(4,1)



list_clusterning_coefs4nodes = clustering_coef(G)

print( list_clusterning_coefs4nodes )



print('Check with built-in function:')

NIdCCfH = snap.TIntFltH()

snap.GetNodeClustCf(G, NIdCCfH)

for item in NIdCCfH:

    print(item, NIdCCfH[item])

    

print()

print()





### Testing  Clustering coef. Erdos-Renyi graph - compare result with built-in function

print('Erdos-Renyi graph')

G = erdos_renyi(5242,14484)

list_clusterning_coefs4nodes = clustering_coef(G, mode_for_end_nodes='put_zero')

print( np.mean(list_clusterning_coefs4nodes) )



print('Compare with built-in function GetNodeClustCf')

NIdCCfH = snap.TIntFltH()

snap.GetNodeClustCf(G, NIdCCfH)

l = [NIdCCfH[item] for item in NIdCCfH ]

print(np.mean(l) )

print('Compare with built-in function GraphClustCoeff')

GraphClustCoeff = snap.GetClustCf (G, -1)

print("Clustering coefficient: %f" % GraphClustCoeff)

import matplotlib.pyplot as plt

import numpy as np



plt.style.use('ggplot')

fig = plt.figure( figsize = (15,6 ) )



for i,graph_type in enumerate(['ErdosRenyi','SmallWorld','CoathorsArxivGrQc']):

    if graph_type == 'ErdosRenyi':

        G = erdos_renyi(5242,14484)

        if 0: # Built-in way for Erdos-Renyi:

            G = snap.GenRndGnm(snap.PNGraph, 5242,14484) # 1000, 10000)

    if graph_type == 'SmallWorld':

        G =  watts_strogatz(5242, 4000  )

    if graph_type == 'CoathorsArxivGrQc':

        G = load_CoathorsArxivGrQc()



    CntV = snap.TIntPrV()

    snap.GetOutDegCnt(G, CntV)

    list_degs = []

    list_counts = []

    for p in CntV:

      #print("degree %d: count %d" % (p.GetVal1(), p.GetVal2()))

      list_degs.append(p.GetVal1() )

      list_counts.append(p.GetVal2() )



    CfVec = snap.TFltPrV()

    snap.GetClustCf(G, CfVec, -1) # https://snap.stanford.edu/snappy/doc/reference/GetClustCf1.html

    list_degs = []

    list_cluster_coef_mean_for_degree = []

    for p in CfVec:

      list_degs.append(p.GetVal1() )

      list_cluster_coef_mean_for_degree.append(p.GetVal2() )



    plt.loglog(list_degs, list_cluster_coef_mean_for_degree, linestyle = 'dotted', label = graph_type)# , color = 'b' 'Collaboration Network')



    plt.xlabel('Node Degree (log)')

    plt.ylabel('Clustering coefficient (mean for nodes with a Given Degree)  (log)')

    plt.title('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')

    plt.legend()



plt.show()



#for pair in CfVec:

#    print("degree: %d, clustering coefficient: %f" % (pair.GetVal1(), pair.GetVal2()))



print('Comment: for small world model we get a declining line - as expected for many natural networks',

      'but actually for real graph of coauthorship we get not very declining line',

        'for Erdos-Renyi we get that coefficient does not depend on degree')