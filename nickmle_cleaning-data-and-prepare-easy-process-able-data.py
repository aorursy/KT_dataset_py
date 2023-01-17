# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/winequality/winequality-red.csv")

df


def splitter(x):

    str(x)

    templist = x.split(';')

    return templist

df.columns[0]

collist = splitter(df.columns[0])

def stringcleaner(x):

    if '"' in x:

        x = x[1:]

        x = x[:-1]

    return x

stringcleaner('"fixed acidity"')
newlist = list()

for i in collist:

    newlist.append(stringcleaner(i))

#print(newlist)

collist = np.array(newlist)

collist
#df.describe()

#df.transpose()

#df
#df = df.trspose()

newdf = pd.DataFrame(columns = collist)

df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'].apply(lambda x : splitter(x))

type(df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'][0])

print(df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'][0])
#x = floatmaker(x[0])

#len(df)

#l = len(df)

newdf.append(x)

df

newdf


for i in range(0, l):

    numstrlist = list()

    numstrlist = floatmaker(splitter(df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'][i]))

    newdf.append(numstrlist, ignore_index=False)
def floatmaker(x):

    flist = x.split(';')

    rlist = list()

    for i in flist:

        rlist.append(float(i))

    return pd.DataFrame(rlist)

a = floatmaker(df['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'][0])

print(type(a[0]), a[0])

#newdf

#col_name = np.arange(len(df.transpose()))

#col_name
newdf = newdf.transpose()

a[0]
def mapper(newdf, df2):

    #newdf

    j = 0

    for i in newdf.columns:

        print(i, df2[0][j])

        newdf[i] = df2[0][j]# assignment operator don't work

        newdf[i]

        

        j+=1

    print(newdf)

    return newdf

newdf = mapper(newdf, a)

newdf
df0 = newdf

#df.columns = col_name
#df = df.transpose()

df.describe()
#columnname = np.arange(len(df))

#columnname

df
df = df.transpose()
len(df)

df.transpose()
df.columns = columnname
import pylab #what this library do is important

import scipy.stats as stats

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline



df[60].unique()
# there are many outliers can encounter when developing the real world application of the 

#

df.corr# correlations between dependent variables == 

# when you learning the technology is important there can be many 

# small change
df
import matplotlib.pyplot as plot

plot.pcolor(df.corr())

plot.show()
import pandas as pd

dfwine = pd.read_csv('../input/winequality/winequality-red.csv')


#fixed acidity the the pH level is low ,, ar 

'''

which file that 

bulb as are no distict pattern -==



'''



dfwine
import networkx as nx

from networkx.algorithms import center

import matplotlib.pyplot as plt
G = nx.Graph()

#nodes = [1,2,3,4, 5]

edges = [(1, 2), (2,3),(1, 3), (3, 4),(4, 1), (5, 2)]

G.add_edges_from(edges)

print(center(G))

G.number_of_edges()
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_color='b',node_size=500, alpha=1)

node_name = {}

nx.draw_networkx_edges(G, pos, width = 8, alpha = 0.5, edge_color='r')

for node in G.nodes():

    node_name[node] = str(node)

nx.draw_networkx_labels(G, pos, node_name, font_size = 16)

plt.axis('off')

plt.show()
from networkx.algorithms.clique import find_cliques, cliques_containing_node
numberofclique = 0

for clique in find_cliques(G):

    print(clique)

    numberofclique +=1

print(numberofclique)
#Here's the grph to look at-

import networkx as nx

G = nx.Graph()

nodes = ["Gur","Qing","Samantha","Jorge","Lakshmi","Jack","John","Jill"]

edges = [("Gur","Qing",{"source":"work"}),

         ("Gur","Jorge", {"source":"family"}),

        ("Samantha","Qing", {"source":"family"}),

        ("Jack","Qing", {"source":"work"}),

        ("Jorge","Lakshmi", {"source":"work"}),

        ("Jorge","Samantha",{"source":"family"}),

        ("Samantha","John", {"source":"family"}),

        ("Lakshmi","Jack", {"source":"family"}),

        ("Jack","Jill", {"source":"charity"}),

        ("Jill","John",{"source":"family"})]

G.add_nodes_from(nodes)

G.add_edges_from(edges)

relationship = {}

for f_edge in edges:

    relationship[f_edge[0:2]] = f_edge[2]

print(relationship)

#for i in range(len(edges)):

 #   relationship.append(edges[i][2])

def get_connections(graph,node,relationship):

    return 0

    ###

    ### YOUR CODE HERE

    ###

G.nodes()

G.edges()
import matplotlib.pyplot as plt
position = nx.spring_layout(G, )

fig = plt.figure(1, figsize=(12, 12))

# here are nodes

#nx.draw_networkx_nodes(G, position, node_color=)

#nx.draw_networkx_edges(G, position, edgelist=G.edges(),width = 15, alpha = 0.5, edge_color='b')

nx.draw_networkx_nodes(G, position, node_size = 0, alpha = 0.8)

nx.draw_networkx_edges(G, position, edgelist = edges, width = 15, alpha = 0.5, edge_color='b')

nx.draw_networkx_edge_labels(G, position, relationship, font_size= 10)

node_name = {}



for node in G.nodes():

    

    node_name[node] = str(node)

#nx.draw_networkx_edge_labels(G, position, relationship, font_size = 16)

nx.draw_networkx_labels(G, position, node_name, font_size = 16)

plt.axis('off')

plt.show()
G.edges()
"""

Modify the following get_connections function so that it takes a network, a person,

and a relationship type as arguments and returns the list of nodes that are connected 

to the person by the relationship. 



Example of use:

get_connections(G,'John','family') 

should return a list of nodes

"""



#Here's the grph to look at-

import networkx as nx

G = nx.Graph()

nodes = ["Gur","Qing","Samantha","Jorge","Lakshmi","Jack","John","Jill"]

edges = [("Gur","Qing",{"source":"work"}),

         ("Gur","Jorge", {"source":"family"}),

        ("Samantha","Qing", {"source":"family"}),

        ("Jack","Qing", {"source":"work"}),

        ("Jorge","Lakshmi", {"source":"work"}),

        ("Jorge","Samantha",{"source":"family"}),

        ("Samantha","John", {"source":"family"}),

        ("Lakshmi","Jack", {"source":"family"}),

        ("Jack","Jill", {"source":"charity"}),

        ("Jill","John",{"source":"family"})]

G.add_nodes_from(nodes)

G.add_edges_from(edges)

relationship = {}

for f_edge in edges:

    relationship[f_edge[0:2]] =f_edge[2]

node = {}

for i in nodes:

    node[i] = str(i)





def get_connections(graph,node,relationship):

    import matplotlib.pyplot as plt

    position = nx.spring_layout(G, )

    fig = plt.figure(1, figsize=(12, 12))

    # here are nodes

    #nx.draw_networkx_nodes(G, position, node_color=)

    #nx.draw_networkx_edges(G, position, edgelist=G.edges(),width = 15, alpha = 0.5, edge_color='b')

    nx.draw_networkx_nodes(G, position, node_size = 0, alpha = 0.8)

    nx.draw_networkx_edges(G, position, edgelist = edges, width = 15, alpha = 0.5, edge_color='b')

    nx.draw_networkx_edge_labels(G, position, relationship, font_size= 10)

        #nx.draw_networkx_edge_labels(G, position, relationship, font_size = 16)

    nx.draw_networkx_labels(G, position, node, font_size = 16)

    plt.axis('off')

    plt.show()

    return 

get_connections(G, node, relationship) 
"""

Modify the following get_connections function so that it takes a network, a person,

and a relationship type as arguments and returns the list of nodes that are connected 

to the person by the relationship. 



Example of use:

get_connections(G,'John','family') 

should return a list of nodes

"""

import networkx as nx

G = nx.Graph()

nodes = ["Gur","Qing","Samantha","Jorge","Lakshmi","Jack","John","Jill"]

edges = [("Gur","Qing",{"source":"work"}),

         ("Gur","Jorge", {"source":"family"}),

        ("Samantha","Qing", {"source":"family"}),

        ("Jack","Qing", {"source":"work"}),

        ("Jorge","Lakshmi", {"source":"work"}),

        ("Jorge","Samantha",{"source":"family"}),

        ("Samantha","John", {"source":"family"}),

        ("Lakshmi","Jack", {"source":"family"}),

        ("Jack","Jill", {"source":"charity"}),

        ("Jill","John",{"source":"family"})]

G.add_nodes_from(nodes)

G.add_edges_from(edges)



def get_connections(graph,node,relationship):

    rlist = list()

    for i in graph.edges():

        if i[2]['source'] != relationship:

            g.remove_edge(i[0], i[1])

    

    newlist = list()

    for c in nx.connected_components(g):

        newlist.append(c)



    for items in newlist:

        if n1 in items:

            for i in items:

                rlist.append(i)

            break   



    return rlist
G.degree
G.edges(nbunch=['John','Gur'], data = True)
g = G

edge1 = edges

G.nodes()
for i in edge1:

    if i[2]['source'] == 'family':

        print(i)

    else:

        g.remove_edge(i[0], i[1])

g.edges()
#nx.connected_components()

print(nx.is_connected(g))

n1 = 'John'

newlist = list()

rlist = list()

for c in nx.connected_components(g):

    newlist.append(c)



for items in newlist:

    if n1 in items:

        for i in items:

            rlist.append(i)

        break

print(rlist)

