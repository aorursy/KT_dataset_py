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
MIN_STATEMENTS = 5
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file_path = "/kaggle/input/kensho-derived-wikimedia-data/property.csv"

p_df = pd.read_csv(file_path)

p_df
file_path = "/kaggle/input/kensho-derived-wikimedia-data/statements.csv"

qpq_df = pd.read_csv(file_path, dtype=np.int)

qpq_df
qpq_source_counts = qpq_df.groupby('source_item_id').size().sort_values(ascending=False)

qpq_source_counts
qpq_source_counts.plot.hist(bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], log=False, ylim=(1e1, 1.75e7))
qpq_source_counts[qpq_source_counts >= MIN_STATEMENTS]
keep_source_item_ids = set(qpq_source_counts[qpq_source_counts >= MIN_STATEMENTS].index)
qpq_df = qpq_df[qpq_df['source_item_id'].isin(keep_source_item_ids)]
qpq_df['source_item_id'].nunique()
p279g = nx.DiGraph()

p279g.add_edges_from(qpq_df[qpq_df['edge_property_id']==279][['source_item_id', 'target_item_id']].values)
root_qids = {

    'per': 5,        # https://www.wikidata.org/wiki/Q5  human

    'loc': 2221906,  # https://www.wikidata.org/wiki/Q2221906  geographic location

    'org': 43229,    # https://www.wikidata.org/wiki/Q43229  organization

    'state': 7275,   # https://www.wikidata.org/wiki/Q7275  state

}
subclass_qids = {

    lbl: set(nx.ancestors(p279g, qid)).union(set([qid]))

    for lbl, qid in root_qids.items()

}
df = pd.DataFrame(index=keep_source_item_ids)

df.index.name = 'qid'
qpq_signature_dfs = {}

mask1 = qpq_df['edge_property_id']==31

for lbl, qid in root_qids.items():

    mask2 = qpq_df['target_item_id'].isin(subclass_qids[lbl])

    qpq_signature_dfs[lbl] = qpq_df[mask1 & mask2][['source_item_id', 'target_item_id']]

    

    qpq_signature_dfs[lbl].set_index('source_item_id', drop=True, inplace=True)

    qpq_signature_dfs[lbl].index.name = 'qid'

    

    # de-duplicate index 

    qpq_signature_dfs[lbl] = qpq_signature_dfs[lbl][~qpq_signature_dfs[lbl].index.duplicated()]

    

    # add to dataframe

    df[lbl] = qpq_signature_dfs[lbl]['target_item_id']
df = df.fillna(0).astype(np.int)

df
mask1 = df['org'] > 0

mask2 = df['state'] > 0

df[mask1 & mask2]
mask1 = df['org'] > 0

mask2 = df['state'] > 0

df[mask2]
def get_most_common_edges(ner_type, qpq_df, p_df):

    ner_qids = df[df[ner_type]>0].index

    common_edges = (

        qpq_df[qpq_df['source_item_id'].isin(ner_qids)].

        groupby('edge_property_id').

        size().

        sort_values(ascending=False).

        to_frame().

        rename(columns={0: 'count'})

    )

    return pd.merge(

        p_df, 

        common_edges, 

        left_on='property_id', 

        right_index=True).sort_values('count', ascending=False)
ner_common_edges = {}

for ner_type in root_qids.keys():

    print(ner_type)

    ner_common_edges[ner_type] = get_most_common_edges(ner_type, qpq_df, p_df)
ner_common_edges['per'].head(50)
ner_common_edges['loc'].head(50)
ner_common_edges['org'].head(50)
ner_common_edges['state'].head(50)