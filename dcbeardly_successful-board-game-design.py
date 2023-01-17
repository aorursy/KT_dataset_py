import plotly.offline

import plotly.graph_objs as go

import networkx as nx

import numpy as np

import pandas as pd

import sqlite3

import re

from itertools import combinations

import matplotlib.pyplot as plt

import seaborn as sns



conn = sqlite3.connect('../input/database.sqlite')



df_games = pd.read_sql_query('SELECT * FROM BoardGames ', conn)



conn.close()



text_columns = []

number_columns = []

other_columns = []



c = list(df_games.columns)



for i in range(df_games.shape[1]):

    if df_games.iloc[:,i].dtype == 'object':

        text_columns.append(c[i])

    elif (df_games.iloc[:,i].dtype == 'float64') or (df_games.iloc[:,i].dtype == 'int64'):

        number_columns.append(c[i])

    else:

        other_columns.append(c[i])

        
print("TEXT COLUMNS:",len(text_columns))

for tcol in text_columns:

    print(tcol)

print('\n')

print("NUMBER COLUMNS:",len(number_columns))

for ncol in number_columns:

    print(ncol)

print('\n')

print("OTHER COLUMNS:",len(other_columns))
text_df = df_games[text_columns]

nmplyrs_df = text_df.iloc[:, 18:29]

print(list(nmplyrs_df))

print('\n')

print(nmplyrs_df.head(10))

print(list(nmplyrs_df['polls.suggested_numplayers.1'].unique()))
mp_dict = {'Best': 1, np.nan: 0, 'Recommended': 1, 'NotRecommended': 0}

remap_df = nmplyrs_df.replace(mp_dict)
num_df = df_games[number_columns]

framesCrrctd = (num_df["stats.bayesaverage"], remap_df)

num_dfCrrctd = pd.concat(framesCrrctd, axis = 1)

print(num_dfCrrctd.corr())
f,ax = plt.subplots(figsize=(13, 12))

sns.heatmap(num_dfCrrctd.corr(), cmap='YlGnBu', annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
def nKey(node):

    se = re.search('(?<=\.)\w+$', node)

    return(se.group(0))

#the regex above reads as:

# "a substring of one or more alphanumeric characters preceded by '.' and followed by the end of the string."

#note that '_' is alphanumeric, but '.' is not.



ndDict = {}

for node in list(num_dfCrrctd):

    ndDict[node] = nKey(node)
G = nx.Graph()

G.add_nodes_from(list(num_dfCrrctd))



df_corr = num_dfCrrctd.corr()        # The correlation matrix for our dataframe



nodelist = list(G.nodes)



uniqueEdges = list(combinations(nodelist, 2))

def weight(edge):

    return(df_corr.loc[edge])



threshold = .5                       # Controls significant correlation threshold for when edges will appear



edgelist = []

for edge in uniqueEdges:

    if weight(edge) >= threshold:

         edgelist.append(edge)



G.add_edges_from(edgelist)



edgelist = list(G.edges)             # Ensures that the order of output is static



weightD={}                           # edges and edge weights as a dict

weightE=[]                           # edge weights as a list

for edge in edgelist:

    weightD[edge] = round(weight(edge),3)

    weightE.append(round(weight(edge),3))



poslist = [(1,8),(2,0),(2,8),(1,6),(2,6),(1,4),(2,4),(1,2),(2,2),(1,0),(1,-2),(3,3)]

#these points are contrived for a better image



pos = dict(zip(sorted(nodelist), poslist))

print(pos)



plt.figure(1,figsize=(13,12))

nx.draw_networkx(G, pos, node_color='violet', node_size=350, labels=ndDict)

nx.draw_networkx_edges(G, pos, edgelist=edgelist, cmap=plt.get_cmap('Accent'), edge_color=weightE)

nx.draw_networkx_edge_labels(G, pos, edge_labels=weightD)



plt.axis('off')

plt.show()