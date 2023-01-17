from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.listdir("../input"))
with open('../input/starwars-full-interactions-allCharacters.json') as f:

    data = json.load(f)

print(data['nodes'][0])

print(data['links'][0])

import networkx as nx

G = nx.Graph()



#build graph nodes

for node in data['nodes']:

    G.add_node(node['name'])

    

#build graph edges

for edge in data['links']:

    G.add_edge(data['nodes'][edge['source']]['name'], data['nodes'][edge['target']]['name'])



# G.add_node((data['nodes'][0]['value'], data['nodes'][0]['name']))

# G.add_nodes_from([(data['nodes'][1]['value'], data['nodes'][1]['name'])])
list(G)

G.number_of_nodes()

G.number_of_edges()
options = {

    'node_color': 'yellow',

    'node_size': 400,

    'width': 1,

    'with_labels': True

}



nx.draw(G, **options)
#breadth-first search of a given graph, given number of layers, and given character

#prints the characters most closely related with the node 'goodSideChar' and its neighbors

def BFS(graph, layers, goodSideChar):

    curNode = goodSideChar

    goodCounts = {curNode: 1}

    visited = {}  #keep track of nodes we have already seen

    queue = []    #keep track of nodes up next



    #copy list of nodes into goodCounts dict

    for node in graph.nodes():

        goodCounts.update({node: 0})

    

    queue.append(curNode)

    visited.update({curNode: True})

    depth = 0

    

    while queue:

        #check if we reached the specified depth

        if queue[0] == 'null':

            depth = depth + 1

            queue.pop(0)

        if layers == depth:

            break

        

        curNode = queue.pop(0)

        for node in list(graph.neighbors(curNode)):

            #if node is already in dict, increment count

            if goodCounts.get(node) is None:

                goodCounts.update({node: 1})

            else:

                goodCounts.update({node: goodCounts.get(node) + 1})

                

            #queue node to be searched

            if visited.get(node) is None:

                visited.update({node: True})

                queue.append(node)

        queue.append('null')    

    

    #print goodCounts in sorted order        

    listofTuples = sorted(goodCounts.items() , reverse=True, key=lambda x: x[1])

    for elem in listofTuples:

        print(elem[0] , ":" , elem[1] )

BFS(G, 1, "OBI-WAN")
cent = nx.degree_centrality(G)

#print cent in sorted order limit 5    

listofTuples = sorted(cent.items() , reverse=True, key=lambda x: x[1])

for i in range(0,5):

    print(listofTuples[i][0] , ":" , listofTuples[i][1] )
bet = nx.betweenness_centrality(G,k=None,normalized=True,weight=None,endpoints=False,seed=None)

#print bet in sorted order limit 5    

listofTuples = sorted(bet.items() , reverse=True, key=lambda x: x[1])

for i in range(0,5):

    print(listofTuples[i][0] , ":" , listofTuples[i][1] )
def calcCliqness(G):

    cliq = nx.find_cliques(G)

    count = 0

    for i in cliq:

        #print(i)

        count += 1

    #print('Count: ' + str(count))



    cliquishness = count / len(G.nodes)

    return cliquishness

    #print(cliquishness)
#flip edges with probability 'prob'

import random as rand

G_Original = G.copy()

prob = .99

count = 0

cliqVals = [calcCliqness(G)]

print('Begining num of edges: ' + str(len(G.edges)))





for e in G.edges:

    if rand.uniform(0,1) <= prob:

        print(str(count) + str(e))

        source = e[0]

        target = data['nodes'][rand.randrange(len(G.nodes))]['name'] 

        while G.has_edge(source, target):

            target = data['nodes'][rand.randrange(len(G.nodes))]['name']

        G.remove_edge(*e)

        G.add_edge(source, target)

        count += 1

        cliqVals.append(calcCliqness(G))

        if(nx.is_connected(G)):

            print(nx.average_shortest_path_length(G))

        else:

            print('Not Connected')



        

print('Ending num of edges: ' + str(len(G.edges)))

print('\nProbability: ' + str(prob))

print('Edges Fliped: ' + str(count))

for i in cliqVals:

    print(i)

    

G = G_Original.copy()
numOfNeighbors = {}



for node in G.nodes():

    count = 0

    for i in G.neighbors(node):

        count += 1

    numOfNeighbors.update({node: count})



#print list of 5 weakest nodes

listofTuples = sorted(numOfNeighbors.items() , reverse=False, key=lambda x: x[1])

for i in range(0,5):

    print(listofTuples[i][0] , ":" , listofTuples[i][1] )
numOfNeighbors = {}



for node in G.nodes():

    count = 0

    for i in G.neighbors(node):

        count += 1

    numOfNeighbors.update({node: count})



#print list of 5 strongest nodes

listofTuples = sorted(numOfNeighbors.items() , reverse=True, key=lambda x: x[1])

for i in range(0,10):

    print(listofTuples[i][0] , ":" , listofTuples[i][1] )