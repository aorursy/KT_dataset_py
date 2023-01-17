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
MIN_STATEMENTS = 1 # 2 # was 5
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
items = pd.read_csv("/kaggle/input/kensho-derived-wikimedia-data/item.csv", nrows = 212345) # first 800k rows - may be enough?

items
# items = items.loc[(~items.en_label.isna() & ~items.en_description.isna())]

items = items.loc[~items.en_label.isna()]

items
items.loc[items["en_label"].str.contains("book",case=True)]
items.loc[items["en_label"].str.contains("film",case=True)]
items.loc[items["en_label"].str.contains("company",case=True)]
items.loc[items["en_label"].str.contains("organization",case=True)]
del (items)
file_path = "/kaggle/input/kensho-derived-wikimedia-data/property.csv"

p_df = pd.read_csv(file_path)

p_df
# p_df.loc[p_df["property_id"]==(31 or 279 or 5 or 2221906 or 43229 or 7275)]
p_df.loc[(p_df["property_id"]==31)|(p_df["property_id"]==279)|(p_df["property_id"]==5)]

### |(p_df["property_id"]==2221906)|(p_df["property_id"]==43229)|(p_df["property_id"]==7275)
file_path = "/kaggle/input/kensho-derived-wikimedia-data/statements.csv"

qpq_df = pd.read_csv(file_path, dtype=np.int) # , nrows=3234567

qpq_df
## keep cases were it's an instance of / is a





qpq_df= qpq_df.loc[(qpq_df["edge_property_id"]==31)|(qpq_df["edge_property_id"]==279)].drop_duplicates()

### |(qpq_df["edge_property_id"]==5)|(qpq_df["edge_property_id"]==2221906)|(qpq_df["edge_property_id"]==43229)|(qpq_df["edge_property_id"]==7275)

print(qpq_df.shape[0])
qpq_df.nunique()
qpq_source_counts = qpq_df.groupby('source_item_id').size().sort_values(ascending=False)



keep_source_item_ids = set(qpq_source_counts[qpq_source_counts >= MIN_STATEMENTS].index)



qpq_source_counts
# qpq_source_counts[qpq_source_counts >= MIN_STATEMENTS]
qpq_df = qpq_df[qpq_df['source_item_id'].isin(keep_source_item_ids)]

# print(qpq_df.shape[0])
%%time

p279g = nx.DiGraph()

## 279  =~ "is instance of"

# p279g.add_edges_from(qpq_df[qpq_df['edge_property_id']==279][['source_item_id', 'target_item_id']].values) ## orig



p279g.add_edges_from(qpq_df[qpq_df['edge_property_id']==279].drop_duplicates()[['source_item_id', 'target_item_id']].values) ## orig

# p279g.add_edges_from(qpq_df[['source_item_id', 'target_item_id']].drop_duplicates().values) ## alt
root_qids = {

    'per': 5,        # https://www.wikidata.org/wiki/Q5  human

    'loc': 2221906,  # https://www.wikidata.org/wiki/Q2221906  geographic location

    'org': 43229,    # https://www.wikidata.org/wiki/Q43229  organization # 	social entity (not necessarily commercial) uni..

    'state': 7275,   # https://www.wikidata.org/wiki/Q7275  state



    "book":571, 

"film":11424,#sequence of images that give the impression of movement

"television_program":15416, #  - segment of audiovisual content intended for broadcast on television

# "company":783794 , ### subclass of organization:  (Q)

    

#     "website":35127, 

 

}
subclass_qids = {

    lbl: set(nx.ancestors(p279g, qid)).union(set([qid]))

    for lbl, qid in root_qids.items()

}
df = pd.DataFrame(index=keep_source_item_ids) ## orig

# df = pd.DataFrame(index=set(qpq_df.index))

df.index.name = 'qid'
qpq_signature_dfs = {}

mask1 = qpq_df['edge_property_id']==31 ### orig

# mask1 = qpq_df['edge_property_id']==(31 or 279) ## alt 



### P31 : that class of which this subject is a particular example and member - is a

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
print(df.shape)

print("old filtering would have given:", df.loc[(df['org'] > 0) |(df['state'] > 0) |(df['loc'] > 0) |(df['per'] > 0)  ].shape[0])
df = df.loc[(df>0).any(1)]  ## is any value over 0 

print(df.shape)
df.to_csv("wikidata_ner_entities_v2.csv.gz",index=True,compression="gzip")