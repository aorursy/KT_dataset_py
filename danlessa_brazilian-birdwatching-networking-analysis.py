import pandas as pd

import matplotlib as mpl

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt
data = pd.read_feather("/kaggle/input/wikiaves_metadata-2019-08-11.feather")



# Try to use less than 2 months for doing debugging,

# the grouping process can take a lot of time

date_start = '2018-01-01'

date_end = '2018-02-01'

data = (data.where(lambda df: ((df['registry_date'] > date_start)

                               & (df['registry_date'] < date_end)))

            .dropna()

            .loc[:, ['author_id', 'registry_date', 'location_id', 'registry_id']])
grouped_data = data.groupby(['registry_date', 'location_id'])

new_data = data.set_index(["registry_date", 'location_id', 'author_id']).copy()

authors = new_data.reset_index().author_id.unique()



N = len(authors)

collab_mtr = np.zeros((N, N))

for i, [name, group] in enumerate(grouped_data):

    print("\r{}/{}    ".format(i, len(grouped_data)), end='')

    new_data.loc[name, 'group'] = i

    arr = group.author_id.unique()

    indices = [np.argwhere(authors == author_id)[0][0] for author_id in arr]

    if len(indices) > 1:

        for ind1 in indices:

            for ind2 in indices:

                collab_mtr[ind1, ind2] += 1

        
G = nx.from_numpy_matrix(collab_mtr)
M = G.number_of_edges()

k = 4 / np.sqrt(G.number_of_nodes())

pos = nx.spring_layout(G, k=k, weight='weight')



node_sizes = np.array([degree[1] for degree in G.degree()])

node_sizes += 1

node_sizes = np.sqrt((node_sizes / node_sizes.max()) + 1) * 10



weights = np.array([w['weight'] for u, v, w in G.edges(data=True)])



plt.figure(figsize=(20, 20), dpi=300)

nx.draw(G, pos,

        node_size=node_sizes,

        node_color=node_sizes,

        node_cmap=plt.cm.viridis,

        width=weights,

        alpha=0.7,

        with_labels=False)

plt.show()