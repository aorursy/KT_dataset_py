# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

file = open("/kaggle/input/cs-data/project2.txt", "r")
nodesInformation={}
edge =[[],[]]
count = 0
fileLine = 0;
for line in file:
    print(line)
    fields = line.split(" - ")
    if (nodesInformation.get(fields[0]) == None):
        nodesInformation[fields[0]] = {'index':count,'nextNode':[],'preNode':[]}
        count+=1
    otherpart = fields[1].split(" (");
    nodesInformation[fields[0]]['courseName'] = otherpart[0]
    connectedlist = otherpart[1].split(")")[0].replace(' and',',').split(', ')
    for connected in connectedlist:
        if (connected!="N/A"):
            if (nodesInformation.get(connected) == None):
                nodesInformation[connected] = {'index':count,'nextNode':[],'preNode':[]}
                count+=1
            edge[0].append(nodesInformation[connected]['index'])
            edge[1].append(nodesInformation[fields[0]]['index'])
            nodesInformation[connected]['nextNode'].append(nodesInformation[fields[0]]['index'])
            nodesInformation[fields[0]]['preNode'].append(nodesInformation[connected]['index'])
    fileLine+=1
#It is good practice to close the file at the end to free up resources   
file.close(),

nodes = []
nodes_index_order = []
for nodeName in nodesInformation:
    nodes.append({'name': nodeName, 'nextNode':nodesInformation[nodeName]['nextNode'],'preNode':nodesInformation[nodeName]['preNode']})
    nodes_index_order.append(nodesInformation[nodeName]['index'])
# sort as low pre nodes number

nodes
for node in nodes:
    if (nodesInformation[node['name']].get('courseName')==None):
        print(node)
# draw function
def drawNetwork(edgelist):
    # Build a dataframe with your connections
    # This time a pair can appear 2 times, in one side or in the other!
    df = pd.DataFrame({ 'from':edgelist[0], 'to':edgelist[1]})
    df

    # Build your graph. Note that we use the DiGraph function to create the graph!
    G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph() )

    # Make the graph
    nx.draw(G, with_labels=True, node_size=500, alpha=0.3, arrows=True, pos=nx.kamada_kawai_layout(G))
drawNetwork(edge)
pre = {}
post = {}
clock = 0
visited={}
def previsit(v):
    global clock
    clock +=1
    pre[v] = clock
def postvisit(v):
    global clock
    clock +=1
    post[v] = clock

def explore(G,v):
    visited[v] = 1
    previsit(v)
    for u in G[v]['nextNode']:
          if not visited[u]:
                explore(G,u)
    postvisit(v)
def dfs(G,order):
    for v in range(0,len(G)):
        visited[v] = 0
    if (order==None):
        order = range(0,len(G))
    for v in order:
        if not visited[v]:
            explore(G,v)
def getTopology():
    post_arr = []
    def takevalue(elem):
        return elem['value']
    for p in post:
        post_arr.append({'key':p,'value':post[p]})
    post_arr.sort(key=takevalue,reverse=True)
    for n in post_arr:
        node = nodes[n['key']]
        print(node['name']+'('+str(len(node['preNode']))+')', end =", ")
dfs(nodes,None)
getTopology()
def sortByNextNodelength (d):
    global nodes
    return len(nodes[d]['nextNode'])
nodes_index_order.sort(key=sortByPreNodelength)
dfs(nodes,nodes_index_order)
getTopology()