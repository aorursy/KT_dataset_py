import numpy as np

import pandas as pd

import igraph

from igraph import Graph

import matplotlib.pylab as plt



import os

for dirname, _, filenames in os.walk('../input/ml-in-graphs-hw0/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
with open('../input/ml-in-graphs-hw0/wiki-Vote.txt') as f:

    with open('removed_comments.txt', 'w') as f1:

        for line in f:

            if line[0] != '#':

                f1.write(line)
path = 'removed_comments.txt'

wiki_graph = Graph.Read_Edgelist(path)
wiki_graph.vcount()
wiki_graph.get_edgelist()[:5]
[i for i in wiki_graph.get_edgelist() if i[0] == i[1]]
wiki_graph.ecount()
edgelist = set(wiki_graph.get_edgelist())



undirected = [i for i in wiki_graph.get_edgelist()[:] if (i[1], i[0]) in edgelist]
len(undirected)/2
wiki_graph.ecount() - len(undirected)/2
has_out = set([i[0] for i in wiki_graph.get_edgelist()])

len([i for i in range(wiki_graph.vcount()) if i not in has_out])
has_in = set([i[1] for i in wiki_graph.get_edgelist()])

len([i for i in range(wiki_graph.vcount()) if i not in has_in])
from collections import Counter
out_counter = Counter([i[0] for i in wiki_graph.get_edgelist()]) 
len([i for i in out_counter.most_common() if i[1] > 10])
in_counter = Counter([i[1] for i in wiki_graph.get_edgelist()]) 

len([i for i in in_counter.most_common() if i[1] < 10])
h = wiki_graph.degree_distribution(bin_width=1, mode='out')
degrees = np.array([i[1] for i in h.bins()])[1:]

n_vs = np.array([i[2] for i in h.bins()])[1:]



degrees = degrees[n_vs>0]

n_vs = n_vs[n_vs>0]
plt.plot(np.log10(degrees), np.log10(n_vs))

plt.grid()
np.polyfit(np.log10(degrees), np.log10(n_vs), 1)
java_graph = Graph.Read_Edgelist('../input/ml-in-graphs-hw0/stackoverflow-Java.txt')
java_graph.vcount()
components = java_graph.components(mode='WEAK')
clustered = components.cluster_graph()
clustered.vcount()
biggest = components.giant()
biggest.ecount(), biggest.vcount()
scores = java_graph.pagerank()
Counter({k:v for k, v in zip(np.arange(java_graph.ecount()), scores)}).most_common()[:3]