import networkx as nx

from PIL import Image

import matplotlib as plt

import pandas as pd

import numpy as np
## We first import our adjacency matrix as a DataFrame. 
df = pd.read_csv('../input/adjacency_matrix_the_wire.csv', index_col='Character')
# We also load the files characters.csv and characters2.csv, which simply contain the information regarding where the characters belong to in the series (The Law, The Streets, Politicians, etc.). I took this characters' division from Wikipedia and thought it could be useful to compare the centrality of characters with those that belong to the same world. This information is in separate files because it would not fit in the adjacency matrix itself.

characters = pd.read_csv('../input/characters.csv')

characters2 = pd.read_csv('../input/characters2.csv')
## Now we import the adjacency matrix to NetworkX for doing our analysis

Graphtype = nx.Graph()

G = nx.from_pandas_adjacency(df)
print(nx.info(G))
## So, we have a total of 65 characters and 298 connections between them. Quite a lot for a fictional series, very impressive storytelling indeed.
degrees = np.array(nx.degree(G))
Characters_Degree = pd.DataFrame(degrees)
Characters_Degree.columns = ['Character', 'Degree']
Characters_Degree['Degree'] = Characters_Degree['Degree'].astype(int)
Characters_Degree['From'] = characters['From']
Characters_Degree.set_index('Character', inplace=True)
Characters_Degree.sort_values(by='Degree', ascending=False).head()
betweenness = nx.betweenness_centrality(G)
Characters_Betweenness = pd.DataFrame.from_records([betweenness], index=[0])
Characters_Betweenness = Characters_Betweenness.transpose()
Characters_Betweenness.columns = ['Betweenness_Centrality']
Characters_Betweenness.index.rename('Character', inplace=True)
centrality = nx.degree_centrality(G)
Characters_Centrality = pd.DataFrame.from_records([centrality], index=[0])
Characters_Centrality = Characters_Centrality.transpose()
Characters_Centrality.columns = ['Degree_Centrality']
Characters_Centrality.index.rename('Character', inplace=True)
closeness = nx.closeness_centrality(G)
Closeness_Centrality = pd.DataFrame.from_records([closeness], index=[0])
Closeness_Centrality = Closeness_Centrality.transpose()
Closeness_Centrality.columns = ['Closeness_Centrality']
Closeness_Centrality.index.rename('Character', inplace=True)
pagerank = nx.pagerank(G)
PageRank_Centrality = pd.DataFrame.from_records([pagerank], index=[0])
PageRank_Centrality = PageRank_Centrality.transpose()
PageRank_Centrality.columns = ['PageRank_Centrality']
PageRank_Centrality.index.rename('Character', inplace=True)
TheWire = pd.concat([Characters_Degree, Characters_Betweenness, Characters_Centrality, Closeness_Centrality, PageRank_Centrality], axis=1)
cols = ['From', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']
TheWire = TheWire[cols]
TheWire.head()
The_Law = TheWire[TheWire['From']=='The Law']
The_Law.sort_values(by='Degree_Centrality', ascending=False, inplace=True)

The_Law[['Degree_Centrality', 'Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]
The_Law.sort_values(by='Betweenness_Centrality', ascending=False, inplace=True)

The_Law[['Betweenness_Centrality', 'Degree', 'Degree_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]
The_Law.sort_values(by='Closeness_Centrality', ascending=False, inplace=True)

The_Law[['Closeness_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'PageRank_Centrality']]
The_Law.sort_values(by='PageRank_Centrality', ascending=False, inplace=True)

The_Law[['PageRank_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']]
The_Street = TheWire[TheWire['From']=='The Street']
The_Street.sort_values(by='Degree_Centrality', ascending=False, inplace=True)

The_Street[['Degree_Centrality', 'Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]
The_Street.sort_values(by='Betweenness_Centrality', ascending=False, inplace=True)

The_Street[['Betweenness_Centrality', 'Degree', 'Degree_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]
The_Street.sort_values(by='Closeness_Centrality', ascending=False, inplace=True)

The_Street[['Closeness_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'PageRank_Centrality']]
The_Street.sort_values(by='PageRank_Centrality', ascending=False, inplace=True)

The_Street[['PageRank_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']]
Politicians = TheWire[TheWire['From']=='Politicians']
Politicians.sort_values(by='Degree_Centrality', ascending=False, inplace=True)

Politicians[['Degree_Centrality', 'Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]
Politicians.sort_values(by='Betweenness_Centrality', ascending=False, inplace=True)

Politicians[['Betweenness_Centrality', 'Degree', 'Degree_Centrality', 'Closeness_Centrality', 'PageRank_Centrality']]
Politicians.sort_values(by='Closeness_Centrality', ascending=False, inplace=True)

Politicians[['Closeness_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'PageRank_Centrality']]
Politicians.sort_values(by='PageRank_Centrality', ascending=False, inplace=True)

Politicians[['PageRank_Centrality', 'Degree', 'Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']]