# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install python-louvain
import matplotlib.pyplot as plt

import numpy as np

import networkx as nx

from networkx.algorithms import community

import random

import pandas as pd

import community as community_louvain

try:

    import pygraphviz

    from networkx.drawing.nx_agraph import graphviz_layout

except ImportError:

    try:

        import pydot

        from networkx.drawing.nx_pydot import graphviz_layout

    except ImportError:

        raise ImportError("This example needs Graphviz and either "

                          "PyGraphviz or pydot")




df_nodes = pd.read_csv('/kaggle/input/stack-overflow-tag-network/stack_network_nodes.csv')

df_edges = pd.read_csv('/kaggle/input/stack-overflow-tag-network/stack_network_links.csv')
df_nodes.head()
df_edges.head()
# get edges and weight

edges = df_edges[['source', 'target']].values.tolist()

weights = [float(l) for l in df_edges.value.values.tolist()]
# Make Graph and apply weight

G = nx.Graph(directed=True)

G.add_edges_from(edges)

for cnt, a in enumerate(G.edges(data=True)):

    G.edges[(a[0], a[1])]['weight'] = weights[cnt]
def simple_Louvain(G):

    """ Louvain method github basic example"""

    partition = community_louvain.best_partition(G)

    pos = graphviz_layout(G)

    

    max_k_w = []

    for com in set(partition.values()):

        list_nodes = [nodes for nodes in partition.keys()

                      if partition[nodes] == com]

        max_k_w = max_k_w + [list_nodes]



    

    node_mapping = {}

    map_v = 0

    for node in G.nodes():

        node_mapping[node] = map_v

        map_v += 1



    community_num_group = len(max_k_w)

    color_list_community = [[] for i in range(len(G.nodes()))]

    

    # color

    for i in G.nodes():

        for j in range(community_num_group):

            if i in max_k_w[j]:

                color_list_community[node_mapping[i]] = j

    

    return G, pos, color_list_community, community_num_group, max_k_w
G, pos, color_list_community, community_num_group, max_k_w = simple_Louvain(G)




edges = G.edges()

Feature_color_sub = color_list_community

node_size = 70



fig = plt.figure(figsize=(20, 10))

im = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=Feature_color_sub, cmap='jet', vmin=0, vmax=community_num_group, with_labels=False)

nx.draw_networkx_edges(G, pos)

nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

plt.xticks([])

plt.yticks([])

plt.colorbar(im)

plt.show(block=False)