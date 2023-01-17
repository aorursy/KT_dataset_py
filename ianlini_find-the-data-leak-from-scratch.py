from functools import partial
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tnrange
train_df = pd.read_csv(
    '../input/santander-value-prediction-challenge/train.csv', index_col='ID')
row_value_counts = [
    {'id': row_id, 'value_counts': row_s[row_s != 0].value_counts()}
    for row_id, row_s in train_df.iterrows()]
row_value_counts[0]
def get_jaccard_index(row_value_count, row_value_count2):
    intersection = (pd.concat(
                        (row_value_count['value_counts'], row_value_count2['value_counts']),
                        axis=1, join='inner')
                    .min(1).sum())
    union = (row_value_count['value_counts'].sum()
             + row_value_count2['value_counts'].sum()
             - intersection)
    return intersection / union

try:
    # The processs runs too long, so let's use the result I generated previously.
    jaccard_index_df = pd.read_hdf('../input/svpc-additional-data/jaccard_index.h5')
except IOError:
    jaccard_index = []
    with ProcessPoolExecutor() as executor:
        for i in tnrange(len(row_value_counts) - 1):
            result = executor.map(partial(get_jaccard_index, row_value_counts[i]),
                                  row_value_counts[i+1:],
                                  chunksize=8)
            jaccard_index.extend(result)
    index = pd.MultiIndex.from_tuples((i, j)
                                      for i in range(len(row_value_counts) - 1)
                                      for j in range(i+1, len(row_value_counts)))
    jaccard_index_df = pd.DataFrame({'jaccard_index': jaccard_index}, index=index)
    jaccard_index_df.to_hdf('jaccard_index.h5', 'df')
jaccard_index_df
jaccard_index_df.describe()
jaccard_index_df['jaccard_index'].hist(bins=20)
threshold = 0.95
pairs = jaccard_index_df.index[jaccard_index_df['jaccard_index'] > threshold].tolist()
print("number of pairs:", len(pairs))
g = nx.Graph()
g.add_edges_from(pairs)
print("number of rows:", len(g))
connected_components = list(nx.connected_components(g))
print("number of groups:", len(connected_components))
biggest_component = max(connected_components, key=len)
# biggest_component = connected_components[2]
nx.draw_networkx(g.subgraph(biggest_component))
rows = [2276, 1327, 2803, 1366, 3901, 2536, 2779, 4309]
same_user_df = train_df.iloc[rows]
same_user_df
def find_feature_pairs(assumed_future: np.ndarray, cols_to_match: np.ndarray):
    is_matched = np.isclose(assumed_future, cols_to_match).all(0)
    return np.where(is_matched)[0]
            
# remove all zero columns
no_all_zeros_same_user_df = (same_user_df.loc[:, ~(same_user_df == 0).all()]
                             .drop(columns='target'))
lag_data = no_all_zeros_same_user_df.iloc[:-1].values
future_data = no_all_zeros_same_user_df.iloc[1:].values
column_pairs = []
for i in range(lag_data.shape[1]):
    matched_idx = find_feature_pairs(lag_data[:, [i]], future_data)
    col_i = no_all_zeros_same_user_df.columns[i]
    column_pairs.extend((col_i, no_all_zeros_same_user_df.columns[idx])
                        for idx in matched_idx)
print("number of pairs:", len(column_pairs))
feature_g = nx.DiGraph()
feature_g.add_edges_from(column_pairs)
print("number of matched features:", len(feature_g))
print("number of groups:", nx.number_weakly_connected_components(feature_g))
# remove the in/out edges of the nodes that have multiple in/out edges
for node in list(feature_g.nodes):
    out_edges = list(feature_g.out_edges(node))
    if len(out_edges) > 1:
        feature_g.remove_edges_from(out_edges)
    in_edges = list(feature_g.in_edges(node))
    if len(in_edges) > 1:
        feature_g.remove_edges_from(in_edges)
# remove isolated nodes
feature_g.remove_nodes_from(list(nx.isolates(feature_g)))

print("number of matched features:", len(feature_g))
components = list(nx.weakly_connected_components(feature_g))
print("number of groups:", len(components))
components.sort(key=len, reverse=True)
Counter(len(c) for c in components)
time_series_features = list(nx.topological_sort(feature_g.subgraph(components[0])))
print(time_series_features)
same_user_df[['target'] + time_series_features]
time_series_features = list(nx.topological_sort(feature_g.subgraph(components[1])))
print(time_series_features)
same_user_df[['target'] + time_series_features]