# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import networkx as nx
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
G_ALL = nx.read_gexf('../input/G_ALL.gexf')
G_AML = nx.read_gexf('../input/G_AML.gexf')
clus_df_ALL = pd.read_csv('../input/clus_df_ALL.csv')
clus_df_AML = pd.read_csv('../input/clus_df_AML.csv')
clus_ALL = {clus_df_ALL['keys'][i]:clus_df_ALL['values'][i] for i in range(clus_df_ALL.shape[0]) }
clus_AML = {clus_df_AML['keys'][i]:clus_df_AML['values'][i] for i in range(clus_df_AML.shape[0]) }
nodes_ALL_1 =  sorted(clus_ALL, key=clus_ALL.get, reverse=True)[:500]
clus_ALL_new = {str(i):clus_ALL[i] for i in nodes_ALL_1}
nodes_AML_1 =  sorted(clus_AML, key=clus_AML.get, reverse=True)[:500]
clus_AML_new = {str(i):clus_AML[i] for i in nodes_AML_1}
G_ALL_new = G_ALL.subgraph(list(clus_ALL_new.keys()))
print(nx.info(G_ALL_new))
node_color = [2*clus_ALL_new[i] for i in list(G_ALL_new.nodes())]
nx.draw_networkx(G_ALL_new, pos=nx.spring_layout(G_ALL_new,scale=2),node_color=node_color,with_labels=False)
from networkx.algorithms.community import greedy_modularity_communities
comm_ALL = list(greedy_modularity_communities(G_ALL_new))
comm_ALL
comm_ALL_ass = {}
for i,c in enumerate(comm_ALL):
    for n in list(c):
        comm_ALL_ass[n] = i
comm_ALL_ass
node_color = [comm_ALL_ass[n] for n in list(G_ALL_new.nodes())]
nx.draw_networkx(G_ALL_new, pos=nx.spring_layout(G_ALL_new,scale=4),node_color=node_color,with_labels=False)
nodelist = [n for n in list(G_ALL_new.nodes()) if comm_ALL_ass[n] in range(4)]
node_color = [comm_ALL_ass[n] for n in nodelist]
nx.draw_networkx(G_ALL_new, pos=nx.spring_layout(G_ALL_new,scale=1),nodelist=nodelist,node_color=node_color,with_labels=False)
G_AML_new = G_AML.subgraph(list(clus_AML_new.keys()))
print(nx.info(G_AML_new))
node_color = [clus_AML_new[i] for i in list(G_AML_new.nodes())]
nx.draw_networkx(G_AML_new, pos=nx.spring_layout(G_AML_new),node_color=node_color,with_labels=False)
comm_AML = list(greedy_modularity_communities(G_AML_new))
comm_AML
comm_AML_ass = {}
for i,c in enumerate(comm_AML):
    for n in list(c):
        comm_AML_ass[n] = i
comm_AML_ass
node_color = [2*comm_AML_ass[n] for n in list(G_AML_new.nodes())]
nx.draw_networkx(G_AML_new, pos=nx.spring_layout(G_AML_new,scale=4),node_color=node_color,with_labels=False)
nodelist = [n for n in list(G_AML_new.nodes()) if comm_AML_ass[n] in range(2)]
node_color = [comm_AML_ass[n] for n in nodelist]
nx.draw_networkx(G_AML_new, pos=nx.spring_layout(G_AML_new,scale=1),nodelist=nodelist,node_color=node_color,with_labels=False)
from networkx.algorithms.core import k_core
G_ALL1 = nx.Graph(G_ALL_new)
G_ALL1.remove_edges_from(nx.selfloop_edges(G_ALL1))
k_core_ALL = k_core(G_ALL1,k=40)
node_color = [comm_ALL_ass[i] for i in list(k_core_ALL.nodes())]
nx.draw_networkx(k_core_ALL,pos=nx.spring_layout(k_core_ALL,scale=0.5),node_color=node_color,with_labels=False)
print(nx.info(k_core_ALL))
from networkx.algorithms.core import k_core
G_AML1 = nx.Graph(G_AML_new)
G_AML1.remove_edges_from(nx.selfloop_edges(G_AML1))
k_core_AML = k_core(G_AML1,k=220)
node_color = [comm_AML_ass[i] for i in list(k_core_AML.nodes())]
nx.draw_networkx(k_core_AML,pos=nx.spring_layout(k_core_AML,scale=1),node_color=node_color,with_labels=False)
print(nx.info(k_core_AML))
