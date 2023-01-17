from collections import Counter

import csv

import gc

import json

import os



import matplotlib.pyplot as plt

import networkx as nx

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm
pd.set_option('max_colwidth', 160)

sns.set()

sns.set_context('talk')
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NUM_STATEMENT_LINES = 141_206_854

NUM_ITEM_LINES = 51_450_317

kdwd_path = os.path.join("/kaggle/input", "kensho-derived-wikimedia-data")
file_path = os.path.join(kdwd_path, "property.csv")

property_df = pd.read_csv(file_path, keep_default_na=False, index_col='property_id')

property_df
property_df.loc[31, 'en_description']
property_df.loc[279, 'en_description']
file_path = os.path.join(kdwd_path, "statements.csv")

chunksize = 1_000_000

qpq_df_chunks = pd.read_csv(file_path, chunksize=chunksize)

qpq_p31_df = pd.DataFrame()

qpq_p279_df = pd.DataFrame()

for qpq_df_chunk in tqdm(qpq_df_chunks, total=NUM_STATEMENT_LINES/chunksize, desc='reading ontology statements'):

    qpq_p31_df = pd.concat([

        qpq_p31_df, 

        qpq_df_chunk[qpq_df_chunk['edge_property_id']==31][['source_item_id', 'target_item_id']]

    ])

    qpq_p279_df = pd.concat([

        qpq_p279_df, 

        qpq_df_chunk[qpq_df_chunk['edge_property_id']==279][['source_item_id', 'target_item_id']]

    ])
# instance of statements

qpq_p31_df
# subclass of statements

qpq_p279_df
keep_p279_ids = (

    set().

    union(set(qpq_p279_df['source_item_id'].values)).

    union(set(qpq_p279_df['target_item_id'].values))

)



keep_p31_ids = (

    set().

    union(set(qpq_p31_df['source_item_id'].values)).

    union(set(qpq_p31_df['target_item_id'].values))

)



keep_item_ids = keep_p279_ids.union(keep_p31_ids)
file_path = os.path.join(kdwd_path, "item.csv")

chunksize = 1_000_000

item_df_chunks = pd.read_csv(

    file_path, chunksize=chunksize, index_col='item_id', keep_default_na=False)

item_df = pd.DataFrame()

for item_df_chunk in tqdm(item_df_chunks, total=NUM_ITEM_LINES/chunksize, desc='reading item labels'):

    item_df = pd.concat([

        item_df, 

        item_df_chunk.loc[set(item_df_chunk.index.values).intersection(keep_item_ids)]

    ])
item_df
item_df[item_df['en_label']=='']
item_df[item_df['en_label']=='city']
is_instance_counts = (

    qpq_p31_df.groupby(['target_item_id']).

    size().

    sort_values(ascending=False).

    to_frame().

    rename(columns={0: 'is_instance_count'})

)
is_instance_counts
is_instance_df = pd.merge(

is_instance_counts,

item_df,

left_index=True,

right_index=True)
is_instance_df
item_id = 61

p31_for_q61 = qpq_p31_df[qpq_p31_df['source_item_id']==item_id]

p31_for_q61
item_df.reindex(p31_for_q61['target_item_id'])
subclass_graph = nx.DiGraph()

subclass_graph.add_edges_from(qpq_p279_df.values)
item_id = 1093829 # city of the United States

in_edges = subclass_graph.in_edges(item_id)

out_edges = subclass_graph.out_edges(item_id)



print('out edges')

print('-' * 20)

for source_item_id, target_item_id in out_edges:

    print('Q{} (label={}, description={})\n is subclass of\nQ{} (label={}, description={})'.format(

        source_item_id,

        item_df.loc[source_item_id, 'en_label'],

        item_df.loc[source_item_id, 'en_description'],

        target_item_id,

        item_df.loc[target_item_id, 'en_label'],

        item_df.loc[target_item_id, 'en_description']))

    print()



print('in edges')

print('-' * 20)

for source_item_id, target_item_id in in_edges:

    print('Q{} (label={}, description={})\n is subclass of\nQ{} (label={}, description={})'.format(

        source_item_id,

        item_df.loc[source_item_id, 'en_label'],

        item_df.loc[source_item_id, 'en_description'],

        target_item_id,

        item_df.loc[target_item_id, 'en_label'],

        item_df.loc[target_item_id, 'en_description']))

    print()
def build_neighborhood(graph, start_qid, k_max):

    subnodes = set([start_qid])

    for k in range(k_max):

        nodes_to_add = set()

        for qid in subnodes:

            nodes_to_add.update(graph.neighbors(qid))

        subnodes.update(nodes_to_add)

    return graph.subgraph(subnodes)
def add_attributes_to_graph(graph, item_df):

    for node in graph:

        graph.nodes[node]['qid'] = node

        graph.nodes[node]['label'] = item_df.loc[node, 'en_label']

        graph.nodes[node]['description'] = item_df.loc[node, 'en_description']

    return graph
def plot_graph(graph):

    fig, ax = plt.subplots(figsize=(14,14))

    pos = nx.circular_layout(graph, scale=2.0)



    node_labels = nx.get_node_attributes(graph, 'label')

    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=18, font_weight=1000)

    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color='red')

    nx.draw_networkx_edges(graph, pos, arrowsize=30, min_target_margin=20)



    xpos = [el[0] for el in pos.values()]

    xmin = min(xpos)

    xmax = max(xpos)

    ypos = [el[1] for el in pos.values()]

    ymin = min(ypos)

    ymax = max(ypos)



    xdif = xmax - xmin

    ydif = ymax - ymin

    fac = 0.3



    ax.set_xlim(xmin-xdif*fac, xmax+xdif*fac)

    ax.set_ylim(ymin-ydif*fac, ymax+ydif*fac)
start_qid = 1093829

k_max = 2

sg = build_neighborhood(subclass_graph, start_qid, k_max)

sg = add_attributes_to_graph(sg, item_df)
plot_graph(sg)