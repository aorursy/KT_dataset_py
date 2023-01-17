# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pyplot as plt

import networkx as nx
G_e = nx.from_edgelist({(1, 2), (2, 1), (1, 3), (1, 1)}, create_using=nx.DiGraph)

A_e = nx.to_numpy_matrix(G_e)
G = nx.read_edgelist('../input/ml-in-graphs-hw0/wiki-Vote.txt', comments='#', delimiter='\t', create_using=nx.DiGraph)

A = nx.to_numpy_matrix(G)
len(G.nodes), len(G), len(G_e)
A.shape[0]
nx.number_of_selfloops(G), nx.number_of_selfloops(G_e)
len([(n, n) for n, nbrs in G.adj.items() if n in nbrs])
np.diagonal(A).sum(), np.diagonal(A_e).sum()
G.size() - nx.number_of_selfloops(G), G_e.size() - nx.number_of_selfloops(G_e)
A.sum() - np.diagonal(A).sum(), A_e.sum() - np.diagonal(A_e).sum()
G.to_undirected().size()  - nx.number_of_selfloops(G), G_e.to_undirected().size()  - nx.number_of_selfloops(G_e)
(A.sum() - np.diagonal(A).sum()) - (np.multiply(np.multiply(A, A.T), (1 - np.eye(A.shape[0]))).sum() / 2)
(A_e.sum() - np.diagonal(A_e).sum()) - (np.multiply(np.multiply(A_e, A_e.T), (1 - np.eye(A_e.shape[0]))).sum() / 2)
G.to_undirected(True).size() - nx.number_of_selfloops(G), G_e.to_undirected(True).size() - nx.number_of_selfloops(G_e)
np.multiply(np.multiply(A, A.T), (1 - np.eye(A.shape[0]))).sum() / 2
np.multiply(np.multiply(A_e, A_e.T), (1 - np.eye(A_e.shape[0]))).sum() / 2
sum([1 for key, out_degree in G.out_degree() if out_degree == 0]), sum([1 for key, out_degree in G_e.out_degree() if out_degree == 0])
(A.sum(axis=1) == 0).sum()
sum([1 for key, in_degree in G.in_degree() if in_degree == 0]), sum([1 for key, in_degree in G_e.in_degree() if in_degree == 0])
(A.sum(axis=0) == 0).sum()
sum([1 for key, out_degree in G.out_degree() if out_degree > 10])
(A.sum(axis=1) > 10).sum()
sum([1 for key, in_degree in G.in_degree() if in_degree < 10])
(A.sum(axis=0) < 10).sum()
_out = np.array([out_degree for key, out_degree in G.out_degree()  if out_degree > 0])

value, count = np.unique(_out, return_counts=True)
plt.figure(figsize=(10, 5))

plt.plot(np.log10(value), np.log10(count), alpha=0.5)

plt.scatter(np.log10(value), np.log10(count))

plt.xlim([np.log10(value).min(), np.log10(value).max()])

plt.xlabel('log10 out-degree')

plt.ylabel('log10 count out-degree')

plt.show()
coef = np.polyfit(np.log10(value), np.log10(count), 1)

print(coef)
plt.figure(figsize=(10, 5))

plt.plot(np.log10(value), np.log10(count), alpha=0.5)

plt.scatter(np.log10(value), np.log10(count))

plt.plot(np.log10(value), np.log10(value) * coef[0] + coef[1], c='r', label='lin reg')

plt.xlim([np.log10(value).min(), np.log10(value).max()])

plt.xlabel('log10 out-degree')

plt.ylabel('log10 count out-degree')

plt.legend()

plt.show()
G = nx.read_edgelist('../input/ml-in-graphs-hw0/stackoverflow-Java.txt', comments='#', delimiter='\t', create_using=nx.DiGraph)
G.size()
len(list(nx.weakly_connected_components(G)))
largest_cc = G.subgraph(max(nx.weakly_connected_components(G), key=len))
largest_cc.size(), len(largest_cc)
%%time

pagerank = nx.pagerank(largest_cc)
{k: v for n, (k, v) in enumerate(sorted(pagerank.items(), key=lambda item: item[1], reverse=True)) if n<3}
%%time

hits = nx.hits(largest_cc)
hubs, authorities = hits
#top hubs

{k: v for n, (k, v) in enumerate(sorted(hubs.items(), key=lambda item: item[1], reverse=True)) if n<3}
#top authorities

{k: v for n, (k, v) in enumerate(sorted(authorities.items(), key=lambda item: item[1], reverse=True)) if n<3}