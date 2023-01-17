from IPython.display import Image

Image("../input/cord19images/GraphDatabase.png")
Image("../input/cord19images/Mulltigraph.png")
# Import libraries

import covid19_tools as cv19

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import networkx as nx

from networkx.algorithms import bipartite

import matplotlib.pyplot as plt
with open(r"/kaggle/input/semantic-similarities-with-use-and-doc2vec/output_data/data.pkl", "rb") as input_file:

    df = pickle.load(input_file)

df.head()
df.columns
#Loading the similarity matrices

use_title_sim_matrix = np.load("/kaggle/input/semantic-similarities-with-use-and-doc2vec/output_data/title_sim.npy")

use_abstract_sim_matrix = np.load("/kaggle/input/semantic-similarities-with-use-and-doc2vec/output_data/abstract_sim.npy")

doc2vec_full_text_sim_matrix = np.load("/kaggle/input/semantic-similarities-with-use-and-doc2vec/output_data/full_text_sim.npy")



#Extracting the articles' ids

index_id = df['cord_uid'].values



#Preparing the dfs that include the ids both as indexes and columns

title_sim_df = pd.DataFrame(use_title_sim_matrix, index = index_id, columns = index_id)

abstract_sim_df = pd.DataFrame(use_abstract_sim_matrix, index = index_id, columns = index_id)

text_sim_df = pd.DataFrame(doc2vec_full_text_sim_matrix, index = index_id, columns = index_id)

title_sim_df.head()
#Creating a dictionary with the nodes (articles)

nodes_dict = dict([x for x in enumerate(index_id)])



#Initialise graph of title similarities

G_title = nx.from_numpy_matrix(np.matrix(use_title_sim_matrix), create_using=nx.Graph) # Creates a graph from a numpy matrix

G_title = nx.relabel_nodes(G_title,nodes_dict) # Relabels the nodes using the Ids

G_title.remove_edges_from(nx.selfloop_edges(G_title)) # Removes selfloops

print("Number of title-graph nodes: {0}, Number of graph edges: {1} ".format(len(G_title.nodes()), G_title.size()))



#Initialise graph of abstract similarities

G_abstract = nx.from_numpy_matrix(np.matrix(use_abstract_sim_matrix), create_using=nx.Graph) # Creates a graph from a numpy matrix

G_abstract = nx.relabel_nodes(G_abstract,nodes_dict) # Relabels the nodes using the Ids

G_abstract.remove_edges_from(nx.selfloop_edges(G_abstract)) # Removes selfloops

print("Number of abstract-graph nodes: {0}, Number of graph edges: {1} ".format(len(G_abstract.nodes()), G_abstract.size()))



#Initialise graph of abstract similarities

G_text = nx.from_numpy_matrix(np.matrix(doc2vec_full_text_sim_matrix), create_using=nx.Graph) # Creates a graph from a numpy matrix

G_text = nx.relabel_nodes(G_text,nodes_dict) # Relabels the nodes using the Ids

G_text.remove_edges_from(nx.selfloop_edges(G_text)) # Removes selfloops

print("Number of text-graph nodes: {0}, Number of graph edges: {1} ".format(len(G_text.nodes()), G_text.size()))
# Check edges of node 'wyz5jyjh'

G_title.edges('wyz5jyjh', data = True)
#Save title-graph to df

df_G_title = nx.to_pandas_edgelist(G_title)

df_G_title = df_G_title.dropna()

df_G_title['sharing'] = 'title_similarity'

print(df_G_title.shape)



#Save abstract-graph to df

df_G_abstract = nx.to_pandas_edgelist(G_abstract)

df_G_abstract = df_G_abstract.dropna()

df_G_abstract['sharing'] = 'abstract_similarity'

print(df_G_abstract.shape)



#Save text-graph to df

df_G_text = nx.to_pandas_edgelist(G_text)

df_G_text = df_G_text.dropna()

df_G_text['sharing'] = 'text_similarity'

print(df_G_text.shape)

df_G_text.head()
df.columns
# Create article-journal df and drop na

article_journal_df = df[['cord_uid', 'journal']].dropna()

print(article_journal_df.shape)

article_journal_df.head()
#Initialise the article journal graph

G_article_journal = nx.Graph()



#Add nodes from df

G_article_journal.add_nodes_from(article_journal_df['cord_uid'].unique(), bipartite = 'articles')

G_article_journal.add_nodes_from(article_journal_df['journal'].unique(), bipartite = 'journals')



#Add edges from df

G_article_journal.add_edges_from(zip(article_journal_df['cord_uid'], article_journal_df['journal']))



print("Number of graph nodes: {0}, Number of graph edges: {1} ".format(len(G_article_journal.nodes()), G_article_journal.size()))
G_article_journal.nodes['wyz5jyjh']
# Check edges of node 'wyz5jyjh'

G_article_journal.edges('wyz5jyjh', data = True)
# Prepare the nodelists needed for computing projections: articles, journals

articles = [n for n in G_article_journal.nodes() if G_article_journal.nodes[n]['bipartite'] == 'articles']

journals = [n for n, d in G_article_journal.nodes(data=True) if d['bipartite'] == 'journals']



# Compute the article projections: articlesG

articlesG = nx.bipartite.projected_graph(G_article_journal, articles)

print("Number of articles graph nodes: {0}, Number of graph edges: {1} ".format(len(articlesG.nodes()), articlesG.size()))
# Calculate the degree centrality using nx.degree_centrality: dcs

dcs = nx.degree_centrality(articlesG)

# Plot the histogram of degree centrality values

plt.hist(list(dcs.values()))

#plt.yscale('log')  

plt.show() 
# Transform graph into a simple graph (not bipartite) and add a weight of 0.2

G_journals = nx.Graph()

for (u, v) in articlesG.edges():

    G_journals.add_edge(u,v,weight=0.2)



print("Number of articles graph nodes: {0}, Number of graph edges: {1} ".format(len(G_journals.nodes()), G_journals.size()))
G_journals.edges('aeogp8c7', data = True)
#Save text-graph to df

df_G_journals = nx.to_pandas_edgelist(G_journals)

#df_G_journals = df_G_text.dropna()

df_G_journals['sharing'] = 'journal'

print(df_G_journals.shape)

df_G_journals.head()
#Append the edgelists dfs

df_G_similarities = df_G_title.append(df_G_abstract, ignore_index=True)

df_G_similarities = df_G_similarities.append(df_G_text, ignore_index=True)

df_G_similarities = df_G_similarities.append(df_G_journals, ignore_index=True)

print(df_G_similarities.shape)

df_G_similarities.head()
# Create Multigraph

M = nx.to_networkx_graph(df_G_similarities, create_using=nx.MultiGraph)

print("Number of multigraph nodes: {0}, Number of graph edges: {1} ".format(len(M.nodes()), M.size()))
# Check edges of node 'wyz5jyjh'

M.edges('wyz5jyjh', data = True)
nx.write_gpickle(M, '/kaggle/working/cord19_multigraph.gpickle')
nbrs1 = M.neighbors('wyz5jyjh')

print(len(list(nbrs1)))
def shared_nodes(G, node1, node2):



    # Get neighbors of node 1: nbrs1

    nbrs1 = G.neighbors(node1)

    # Get neighbors of node 2: nbrs2

    nbrs2 = G.neighbors(node2)



    # Compute the overlap using set intersections

    overlap = set(nbrs1).intersection(nbrs2)

    return overlap



#Check the number of shared nodes between the first and second article

print(len(shared_nodes(M, 'wyz5jyjh', 'aeogp8c7')))