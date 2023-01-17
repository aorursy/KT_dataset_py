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
import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
G=nx.Graph()
G.add_node(1)

G.add_node(2)

G.add_node(3)

G.add_node(4)

G.add_node(5)

G.add_node(6)
G.nodes()
G.add_edge(1,5)

G.add_edge(1,3)

G.add_edge(4,6)

G.add_edge(5,4)

G.add_edge(2,3)

G.add_edge(6,3)

G.add_edge(2,4)



# Or we can give a list of edges as input 

#g.add_edge_from([(1,5),(1,3),(4,6),(5,4),(2,3),(6,3),(2,4)])
G.edges()
print(nx.info(G))
G.node[1]['label']='Blue'

G.node(data=True)
import matplotlib.pyplot as plt 

nx.draw(G,with_labels=1)

plt.show()

plt.ioff()
G.node[1]['label']='Blue'

G.node(data=True)
print(nx.diameter(G))
Z=nx.complete_graph(10)
Z.nodes()
Z.edges()
Z.order()
Z.size()
print(nx.info(Z))
nx.draw(Z,with_labels=1)

plt.show()
H=nx.complete_graph(100)

nx.draw(H)
G=nx.gnp_random_graph(20,0.5)

nx.draw(G)

plt.show()
# Creating Graph object

g=nx.Graph() 



#Creating edges

g.add_edge('Bumrah','Shami')

g.add_edge('Bumrah','Kumar')

g.add_edge('Chahal','Kuldeep')

g.add_edge('Kolhi','Bumrah')

g.add_edge('Kuldeep','Kolhi')

g.add_edge('Rohit','Rahul')

g.add_edge('Rohit','Kolhi')

g.add_edge('Rohit','Maynak')

g.add_edge('Dhoni','Pant')

g.add_edge('Dhoni','Karthik')

g.add_edge('Dhoni','Kolhi')

g.add_edge('Pandya','Kedar')

g.add_edge('Pandya','Jadeja')

g.add_edge('Pandya','Kolhi')



#Defining the figure Size

plt.rcParams['figure.figsize']=(10,10)

plt.style.use('fivethirtyeight')





pos=nx.spring_layout(g)



# Drawing Nodes 

nx.draw_networkx_nodes(g,pos,node_size=900,node_color='Orange')



# Drawing Edges

nx.draw_networkx_edges(g,pos,width=6,alpha=0.7,edge_color='Blue')



# Drawing Labels

nx.draw_networkx_labels(g,pos,font_size=20,font_family='sans-serif')

plt.axis('off')

plt.show()

# Directed Graphs



g = nx.DiGraph()

g.add_edge('A', 'B')

g.add_edge('B', 'C')

g.add_edge('C', 'H')

g.add_edge('B', 'D')

g.add_edge('A', 'E')

g.add_edge('A', 'F')

g.add_edge('A', 'G')



import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (10, 10)

plt.style.use('fivethirtyeight')



pos = nx.spring_layout(g)



# drawing nodes

nx.draw_networkx_nodes(g, pos, node_size = 900, node_color = 'yellow')



# drawing edges

nx.draw_networkx_edges(g, pos, edge_color = 'brown', width = 6, alpha = 0.5)



# defining labels

nx.draw_networkx_labels(g, pos, font_size=20, font_family='sans-serif')



plt.title('Directed Graphs', fontsize = 20)

plt.axis('off')

plt.show()
import networkx as nx

import pandas as pd

import csv

import matplotlib.pyplot as plt

%matplotlib inline
# We are using data from directed graph to find closness for Node A

g.degree("A")
# Degree of all the other nodes in directed graph g

g.degree()
# Here's the top 5.

sorted(g.degree(), key=lambda x:x[1], reverse=True)[:5]
# Degree for the 'A' node

degree_A = g.degree("A")  # 4 romantic partners



# Total number of nodes (excluding Grey) 

total_nodes_minus_A = len(g.nodes())-1  # 31 characters in the cast, excluding A



# Degree centrality for A

degree_centrality_A = (degree_A / total_nodes_minus_A)

print("Calculated degree centrality for A:", degree_centrality_A)



# Double check

print("Networkx degree centrality for A:", nx.degree_centrality(g)["A"])



def check_equal(val1, val2):

    assert (val1 == val2),"Centrality measure calculated incorrectly!"

    return "Values match, good job!"



check_equal(degree_centrality_A, nx.degree_centrality(g)["A"])
degree_centrality = nx.degree_centrality(g)

degree_centrality
# Top 5.  Percent of cast this character has been with.

sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]