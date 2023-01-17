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
# For Kaggle: Do not forget to TURN ON INTERNET in kaggle SETITTINGs (right panel)

!python -m pip install snap-stanford
import snap
# Based on: https://snap.stanford.edu/snappy/doc/tutorial/tutorial.html

Gsmall = snap.TNGraph.New() # Create Empty ORIENTED graph

Gsmall.AddNode(1) # Add nodes 

Gsmall.AddNode(2)

Gsmall.AddNode(3)

# (1, 2), (2, 1), (1, 3), (1, 1)}.

Gsmall.AddEdge(1,2)

Gsmall.AddEdge(2,1)

Gsmall.AddEdge(1,3)

Gsmall.AddEdge(1,1)



for NI in Gsmall.Nodes():

  print("node: %d, out-degree %d, in-degree %d" % ( NI.GetId(), NI.GetOutDeg(), NI.GetInDeg()))



G = Gsmall

print('Nodes count', G.GetNodes())

print('Edges count', G.GetEdges() )

Count = snap.CntSelfEdges(G)

print("Directed Graph: Count of self edges is %d" % Count) # https://snap.stanford.edu/snappy/doc/reference/CntSelfEdges.html

Count = snap.CntUniqDirEdges(G)

print("Directed Graph: Count of unique directed edges is %d" % Count) # https://snap.stanford.edu/snappy/doc/reference/CntUniqDirEdges.html

Count = snap.CntUniqUndirEdges(G)

print("Directed Graph: Count of unique undirected edges is %d" % Count) # https://snap.stanford.edu/snappy/doc/reference/CntUniqUndirEdges.html



snap.DrawGViz(Gsmall, snap.gvlDot, "Gsmall_grid5x3.png", "Grid 5x3") # Save image to png

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img = mpimg.imread('Gsmall_grid5x3.png')

plt.imshow(img)

plt.show()



filepath = "/kaggle/input/ml-in-graphs-hw0/wiki-Vote.txt"

import os

print('File exists: ',  os.path.isfile(filepath  )  )



G1 = snap.LoadEdgeList(snap.PNGraph,filepath,0,1)

    # Load graph presented by edge list, i.e. pairs of number edge-source/edge-destination

    # Help: https://snap.stanford.edu/snappy/doc/reference/LoadEdgeList.html

    # snap.PNGraph - means ORIENTED graph, other options: #PUNGraph, an undirected graph;   #PNEANet, a directed network;

    # 0,1, - means taking data from the columns 0 and 1
G1.GetNodes()
# Implementation 1:

result = 0

for EI in G1.Edges():

    if EI.GetSrcNId()==EI.GetDstNId():

        print ("(%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))

        result+=1

print(result)



# Implementation 2:

Count = snap.CntSelfEdges(G1)

print("Directed Graph: Count of self edges is %d" % Count) # https://snap.stanford.edu/snappy/doc/reference/CntSelfEdges.html





# Implementation 1

result = 0

for EI in G1.Edges():

    if EI.GetSrcNId()!=EI.GetDstNId():

        result+=1

print(result)



# Implementation 2

Count = snap.CntUniqDirEdges(G1)

print("Directed Graph: Count of unique directed edges is %d" % Count) # https://snap.stanford.edu/snappy/doc/reference/CntUniqDirEdges.html



# Implementation 

Count = snap.CntUniqUndirEdges(G1)

print("Directed Graph: Count of unique undirected edges is %d" % Count) # https://snap.stanford.edu/snappy/doc/reference/CntUniqUndirEdges.html

# Implementation 1



print(snap.CntUniqDirEdges(G1) - snap.CntUniqUndirEdges(G1))





# Implementation 2 - INCORRECT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1



# Beware that function GetEdgesInOut which is described as giving "Returns the number of reciprocal edges"

# cannot be used here



# Create vector of nodes IDs - should be more simple way ( !? )

Nodes = snap.TIntV()

for NI in G1.Nodes():

  Nodes.Add(NI.GetId() )

results = snap.GetEdgesInOut(G1, Nodes) # https://snap.stanford.edu/snappy/doc/reference/GetEdgesInOut.html

print("Incorrect : %s" % (results[0]))  

# Implementation 1

result = 0

for node in G1.Nodes():

    out_set = set()

    for edge in node.GetOutEdges():

        out_set.add(edge)

    if len(out_set)==0:

        result+=1

print(result)



# Implementation 2

Count = snap.CntOutDegNodes(G1, 0)

print("Directed Graph: Count of nodes with out-degree 0 is %d" % Count)
# Implementation 1

result = 0

for node in G1.Nodes():

    in_set = set()

    for edge in node.GetInEdges():

        in_set.add(edge)

    if len(in_set)==0:

        result+=1

print(result)



# Implementation 2

Count = snap.CntInDegNodes(G1, 0)

print("Directed Graph: Count of nodes with in-degree 0 is %d" % Count)



# Implementation 1

result = 0

for node in G1.Nodes():

    out_set = set()

    for edge in node.GetOutEdges():

        out_set.add(edge)

    if len(out_set)>10:

        result+=1

print(result)



# Implementation 2

result_in_degree = snap.TIntV() # Prepare vector where IN-degrees will be stored

result_out_degree = snap.TIntV() # Prepare vector where OUT-degrees will be stored



snap.GetDegSeqV(G1, result_in_degree, result_out_degree) # Get vectors with all degrees (for ALL nodes at once)

# https://snap.stanford.edu/snappy/doc/reference/GetDegSeqV.html



print('Number of nodes with OUT-degree greater 10:',  (np.array(result_out_degree) > 10).sum()  )





# Implementation 1



result = 0

for node in G1.Nodes():

    in_set = set()

    for edge in node.GetInEdges():

        in_set.add(edge)

    if len(in_set)<10:

        result+=1

print(result)



# Implementation 2

result_in_degree = snap.TIntV() # Prepare vector where IN-degrees will be stored

result_out_degree = snap.TIntV() # Prepare vector where OUT-degrees will be stored



snap.GetDegSeqV(G1, result_in_degree, result_out_degree) # Get vectors with all degrees (for ALL nodes at once)

# https://snap.stanford.edu/snappy/doc/reference/GetDegSeqV.html



print('Number of nodes with IN-degree greater 10:', (np.array(result_in_degree) < 10).sum()  )

# Implementation 1



result = []

for node in G1.Nodes():

    out_set = set()

    for edge in node.GetOutEdges():

        out_set.add(edge)

    result.append(len(out_set))

df = pd.DataFrame(result).groupby(0).size()

df = df[1:]

x = np.log(df.index)

y = np.log(df.values)

import matplotlib.pyplot as plt

plt.bar(x, y)

plt.show()

from scipy.stats import linregress

linregress(x, y)



# Implementation 2 

label = 'OUT degrees'

vec_degrees = result_out_degree

mx = np.max( vec_degrees )

bins = np.arange(1,mx) # Avoid zero - since it is outlier

h = np.histogram( vec_degrees , bins = bins)# [1,2,3,4,5,6,7,9,10,11,12,13,14,15] )

#plt.loglog(h[1][:-1], h[0],'*-', label = label, linewidth=4)

#plt.legend()

#plt.show()

vec_degrees = result_out_degree

mx = np.max( vec_degrees )

bins = np.arange(1,mx) # Avoid zero - since it is outlier

h = np.histogram( vec_degrees , bins = bins)# [1,2,3,4,5,6,7,9,10,11,12,13,14,15] )

m = h[0] > 0   # Need to avoid zeros, before taking LOG

x = np.log10( h[1][:-1][m] ) # Bins

y = np.log10( h[0][m] ) # Counts 



coefs_polyfit = np.polyfit(x, y, 1)

vec_line_intepolation_result = np.poly1d(coefs_polyfit)(x)

print( 'coefficients of line interpolation: ', coefs_polyfit )

plt.style.use('ggplot') # will create nicer plots automatically - grid, etc...

fig = plt.figure(figsize= (10,4) )

plt.loglog(h[1][:-1], h[0],'*-', label = label, linewidth=4)

plt.loglog(np.power(10,x), np.power(10, vec_line_intepolation_result ) ,  label = label+' Approx', linewidth=4)

plt.title('LogLog plot')

plt.xlabel('degree')

plt.ylabel('Node count')

plt.legend()

plt.show()

G2 = snap.LoadEdgeList(snap.PNGraph,"/kaggle/input/ml-in-graphs-hw0/stackoverflow-Java.txt",0,1)

print('Nodes count', G2.GetNodes())

print('Edges count', G2.GetEdges() )
# Weakly connected component for DIRECTED graph - connected component made which is connected by edges ignoring their direction 

# https://mathworld.wolfram.com/WeaklyConnectedComponent.html



# Strongly connected component - any two nodes can be connected by path RESPECTING direction

#https://en.wikipedia.org/wiki/Strongly_connected_component



Components = snap.TCnComV()

snap.GetWccs(G2, Components)

print( len(Components) )

MxWcc = snap.GetMxWcc(G2)

MxWcc.GetEdges(),MxWcc.GetNodes()
result = []

PRankH = snap.TIntFltH()

snap.GetPageRank(G2, PRankH) # https://snap.stanford.edu/snappy/doc/reference/GetPageRank.html

for item in PRankH: # Is there a way without loop ? 

    result.append([item, PRankH[item]])

result.sort(key= lambda x: x[1],reverse=True)

print('Top 3 nodes: ', [x[0] for x in result[:3]] )

print('Their Page ranks: ', [x[1] for x in result[:3]] )

NIdHubH = snap.TIntFltH()

NIdAuthH = snap.TIntFltH()

snap.GetHits(G2, NIdHubH, NIdAuthH) # https://snap.stanford.edu/snappy/doc/reference/GetHits.html

result = []

for item in NIdHubH:

    result.append([item, NIdHubH[item]])

    

result.sort(key= lambda x: x[1],reverse=True)

print('Top 3 HUB nodes: ', [x[0] for x in result[:3]] )

print('Their HITS ranks: ', [x[1] for x in result[:3]] )

result = []

for item in NIdAuthH:

    result.append([item, NIdAuthH[item]])

    

result.sort(key= lambda x: x[1],reverse=True)

print('Top 3 AUTHORITY nodes: ', [x[0] for x in result[:3]] )

print('Their HITS ranks: ', [x[1] for x in result[:3]] )

G = snap.GenGrid(snap.PUNGraph, 5, 3) # Create Test Graph 

snap.DrawGViz(G, snap.gvlDot, "grid5x3.png", "Grid 5x3") # Save image to png

import os

os.listdir() # Check file was saved 
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img = mpimg.imread('grid5x3.png')

plt.imshow(img)

plt.show()