import numpy as np

import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline

# read in the person-to-person CSV file. Drop a column. Check that it looks good.

df = pd.read_csv("../input/Person_Person.csv", encoding = "ISO-8859-1")

df.drop('Source(s)', inplace=True, axis=1)

df.head()
# create networkx graph

G = nx.Graph()
# add nodes for all of the people in the list

for i in range(len(df)):

    G.add_node(df[['Person A', 'Person B']].iloc[i][0])

    G.add_node(df[['Person A', 'Person B']].iloc[i][1])
# add edges

for i in range(len(df)):

    G.add_edge(df[['Person A', 'Person B']].iloc[i][0],df[['Person A', 'Person B']].iloc[i][1])
# draw the plot, its a little large, but I like to see the labels clearly

# even so, its still a little crowded.

plt.figure(figsize=(25,25))

plt.axis('equal')

plt.title('person-to-person')

nx.draw(G, node_color='b', node_size=50, with_labels=True)