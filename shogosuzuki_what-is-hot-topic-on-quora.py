%matplotlib inline

import numpy as np

import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")
df.head(3)
df_duplicate = df[df.is_duplicate == 1]

df_unduplicate = df[df.is_duplicate == 0]

duplicated_edges = df_duplicate[["qid1", "qid2"]].values

unduplicated_edges = df_unduplicate[["qid1", "qid2"]].values

duplicated_graph = nx.Graph()

duplicated_graph.add_edges_from(duplicated_edges)

unduplicated_graph = nx.Graph()

unduplicated_graph.add_edges_from(unduplicated_edges)
def draw_degree_distribution(G):

    hist = nx.degree_histogram(G)

    print(hist)

    plt.scatter(np.arange(len(hist)), hist)

    plt.xlabel("Degree (d)")

    plt.ylabel("Frequency")
# is_duplicate == 1

draw_degree_distribution(duplicated_graph)
# is_duplicate == 0

draw_degree_distribution(unduplicated_graph)
duplicated_graph_degree = [(k, v) for k, v in duplicated_graph.degree().items()]

duplicated_graph_degree.sort(key=lambda p: p[1], reverse=True)
duplicated_graph_degree[:10]
df_28133 = df_duplicate[(df_duplicate.qid1 == 28133) | (df_duplicate.qid2 == 28133)]
df_28133.head()
qid = 3595

df_tmp = df_duplicate[(df_duplicate.qid1 == qid) | (df_duplicate.qid2 == qid)]

df_tmp.head()
qid = 38

df_tmp = df_duplicate[(df_duplicate.qid1 == qid) | (df_duplicate.qid2 == qid)]

df_tmp.head()
df_tmp.head().question1.values
qid = 4951

df_tmp = df_duplicate[(df_duplicate.qid1 == qid) | (df_duplicate.qid2 == qid)]

df_tmp.head()