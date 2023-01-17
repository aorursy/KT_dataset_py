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
G1 = snap.LoadEdgeList(snap.PNGraph,"/kaggle/input/ml-in-graphs-hw0/wiki-Vote.txt",0,1)
# nodes = set()
# G1 = snap.TNGraph.New()
# for line in ['1\t2', '2\t1', '1\t3', '1\t1']:
#     node1, node2 = line.replace('\n','').split('\t')
#     node1 = int(node1)
#     node2 = int(node2)
#     if node1==node2:
#         print(node1)
#     if node1 not in nodes:
#         G1.AddNode(node1)
#         nodes=set.union(nodes,{node1})
#     if node2 not in nodes:
#         G1.AddNode(node2)
#         nodes=set.union(nodes,{node2})
#     G1.AddEdge(node1,node2)
G1.GetNodes()
result = 0
for EI in G1.Edges():
    if EI.GetSrcNId()==EI.GetDstNId():
        print ("(%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
        result+=1
print(result)
result = 0
for EI in G1.Edges():
    if EI.GetSrcNId()!=EI.GetDstNId():
        result+=1
print(result)
result = 0
edges = 0
for node in G1.Nodes():
    node_id = node.GetId()
    in_set = set()
    for edge in node.GetInEdges():
        if node_id!=edge:
            in_set.add(edge)
            edges+=1
    out_set = set()
    for edge in node.GetOutEdges():
        if node_id!=edge:
            out_set.add(edge)
            edges+=1
    result+=len(set.intersection(out_set, in_set) )
print(edges/2-result/2)
print('Undirected edges amount: {}'.format(snap.CntUniqUndirEdges(G1)))
result = 0
Nodes = snap.TIntV()
for nodeId in G1.Nodes():
    Nodes.Add(nodeId.GetId())
    
for node in G1.Nodes():
    node_id = node.GetId()
    in_set = set()
    for edge in node.GetInEdges():
        if node_id!=edge:
            in_set.add(edge)
    out_set = set()
    for edge in node.GetOutEdges():
        if node_id!=edge:
            out_set.add(edge)
    result+=len(set.intersection(out_set, in_set) )
print(result/2)

print('Reciprocated edges amount: {}'.format(snap.CntUniqDirEdges(G1) - snap.CntUniqUndirEdges(G1)))

# results = snap.GetEdgesInOut(G1,Nodes)
result = 0
for node in G1.Nodes():
    out_set = set()
    for edge in node.GetOutEdges():
        out_set.add(edge)
    if len(out_set)==0:
        result+=1
print(result)
result = 0
for node in G1.Nodes():
    in_set = set()
    for edge in node.GetInEdges():
        in_set.add(edge)
    if len(in_set)==0:
        result+=1
print(result)
result = 0
for node in G1.Nodes():
    out_set = set()
    for edge in node.GetOutEdges():
        out_set.add(edge)
    if len(out_set)>10:
        result+=1
print(result)
result = 0
for node in G1.Nodes():
    in_set = set()
    for edge in node.GetInEdges():
        in_set.add(edge)
    if len(in_set)<10:
        result+=1
print(result)
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
from scipy.stats import linregress
linregress(x, y)
G2 = snap.LoadEdgeList(snap.PNGraph,"/kaggle/input/ml-in-graphs-hw0/stackoverflow-Java.txt",0,1)
Components = snap.TCnComV()
snap.GetWccs(G2, Components)
len(Components)
MxWcc = snap.GetMxWcc(G2)
MxWcc.GetEdges(),MxWcc.GetNodes()
result = []
PRankH = snap.TIntFltH()
snap.GetPageRank(G2, PRankH)
for item in PRankH:
    result.append([item, PRankH[item]])
result.sort(key= lambda x: x[1],reverse=True)
[x[0] for x in result[:3]]
NIdHubH = snap.TIntFltH()
NIdAuthH = snap.TIntFltH()
snap.GetHits(G2, NIdHubH, NIdAuthH)
result = []
for item in NIdHubH:
    result.append([item, NIdHubH[item]])
    
result.sort(key= lambda x: x[1],reverse=True)
[x[0] for x in result[:3]]
result = []
for item in NIdAuthH:
    result.append([item, NIdAuthH[item]])
    
result.sort(key= lambda x: x[1],reverse=True)
[x[0] for x in result[:3]]