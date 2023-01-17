# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import networkx as nx

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/FacebookRecruiting/train.csv')

data
data.groupby('source_node')['source_node'].value_counts().sort_values()
data.isna().any().sum()
data[['source_node', 'destination_node']]
Graphtype = nx.DiGraph()

G = nx.from_pandas_edgelist(data.head(), source = 'source_node', target = 'destination_node', create_using=Graphtype)
pos=nx.spring_layout(G)

nx.draw(G,pos,node_color='#87ceeb',edge_color='#fc0362', with_labels=True)

print(nx.info(G))
data['source_node'].nunique()
data['source_node'].nunique() / data.shape[0]
H = G.to_undirected()

for i in nx.ego_graph(H, 1, radius=2, undirected=False).edges:

    print(nx.jaccard_coefficient(H, i))

    #list.append(tuple(i))
for u, v, p in nx.jaccard_coefficient(H, nx.ego_graph(H, 1, radius=2, undirected=False).edges):

    print(p)