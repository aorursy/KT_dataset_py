import pandas as pd

import numpy as np

import random

import pathlib

import gc

import os

import functools

import collections

import pickle



import networkx as nx

from tqdm import tqdm

tqdm.pandas()
DEBUG = False



RANDOM_SEED = 43

CACHE_OUT_FOLDER = pathlib.Path('../working/cache')

CACHE_IN_FOLDER = pathlib.Path('../input/social-task-v1/cache')

DROP_FIXED_DUPLICATE = True

NEG_EDGES_MULTIPLIER = 50



random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)
if not os.path.exists(CACHE_OUT_FOLDER):

    os.makedirs(CACHE_OUT_FOLDER)
print('Reading adjancencies')

with open(CACHE_IN_FOLDER / 'adjs.npy', 'rb') as f:

    pairs_np = np.load(f)



print("Reading edge features")

with open(CACHE_IN_FOLDER / 'edge_features.npy', 'rb') as f:

    edge_features = np.load(f)



print("Reading metrics")

with open(pathlib.Path(CACHE_IN_FOLDER / 'metrics.npy'), 'rb') as f:

    metrics = np.load(f)



print("Reading target")

with open(CACHE_IN_FOLDER / 'target.npy', 'rb') as f:

    target = np.load(f)
num_pairs = pairs_np.shape[0]



uniqued_pairs = np.full((num_pairs // 2, pairs_np.shape[1]), -1, dtype='int32')

uniqued_metrics = np.full((num_pairs // 2, metrics.shape[1]), -1, dtype='float32')

uniqued_target = np.full((num_pairs // 2), -1, dtype='int8')



edge_features_num = edge_features.shape[1]



# Uniqued edge features is 2 times wider than the original edge features,

# because we can't just discard it, like metrics.

# These features are directed, e.g. edge_features(a, b) != edge_features(b, a).

uniqued_edge_features = np.full((num_pairs // 2, edge_features_num * 2), -1, dtype='float32')



seen = {}

idx = 0

for pair, p_metric, p_edge_feature, p_target in tqdm(zip(pairs_np, metrics, edge_features, target), total=len(pairs_np)):

    sorted_pair = tuple(sorted(pair))

    if sorted_pair not in seen:

        uniqued_pairs[idx] = sorted_pair

        uniqued_metrics[idx] = p_metric

        uniqued_target[idx] = p_target

        uniqued_edge_features[idx][:edge_features_num] = p_edge_feature

        seen[sorted_pair] = idx

        idx += 1

    else:

        seen_idx = seen[sorted_pair]

        uniqued_edge_features[seen_idx][edge_features_num:] = p_edge_feature
del pairs_np, metrics, edge_features, seen, target

gc.collect()
if DEBUG:

    uniqued_pairs = uniqued_pairs[:50000]

    uniqued_metrics = uniqued_metrics[:50000]

    uniqued_edge_features = uniqued_edge_features[:50000]

    uniqued_target = uniqued_target[:50000]
edge_features = pd.DataFrame(uniqued_edge_features)

edge_features.head()
def get_duplicate_columns(df):

    '''

    Get a list of duplicate columns.

    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.

    :param df: Dataframe object

    :return: List of columns whose contents are duplicates.

    '''

    duplicateColumnNames = set()

    # Iterate over all the columns in dataframe

    for x in range(df.shape[1]):

        # Select column at xth index.

        col = df.iloc[:, x]

        # Iterate over all the columns in DataFrame from (x+1)th index till end

        for y in range(x + 1, df.shape[1]):

            # Select column at yth index.

            otherCol = df.iloc[:, y]

            # Check if two columns at x 7 y index are equal

            if col.equals(otherCol):

                duplicateColumnNames.add(df.columns.values[y])

 

    return list(duplicateColumnNames)
duplicate_columns = get_duplicate_columns(edge_features.iloc[:1000, :])

print(f'Found duplicate columns: {duplicate_columns}')

edge_features = edge_features.drop(columns=duplicate_columns)
all_features = pd.DataFrame(uniqued_pairs).rename(columns={0: 'id1', 1: 'id2'})

all_features = pd.concat(

    (

        all_features,

        pd.DataFrame(uniqued_metrics).rename(columns={0: 'CN', 1: 'AA', 2: 'RA'}),

        edge_features.add_suffix('_feat'),

        pd.Series(uniqued_target, name='target')

    ), axis=1)

all_features.head()



print(f'Отсутствуют -1 (дефолтное значение при np.full): {(all_features != -1).all(axis=None)}')

print(f'Отсутствуют NaN: {(~all_features.isna()).all(axis=None)}')



del uniqued_metrics, edge_features, uniqued_target, uniqued_pairs

gc.collect()
all_features['target'] = all_features['target'].astype('bool')
def compute_metrics(g, metrics, node_pair):

    node1, node2 = node_pair

    result = np.zeros(len(metrics), dtype='float32')



    for i, metric in enumerate(metrics):

        result[i] = np.round(metric(g, node1, node2), 3)



    return result



def compute_features(g, feature_agg, node_pair):

    node1, node2 = node_pair

    res = [agg(feature(g, node1, node2)) for feature, agg in feature_agg]

    return res



def common_neighbors_score(g, node1, node2):

    common_n = _common_neighbors(g, node1, node2)

    return common_n.shape[0]



@functools.lru_cache()

def _common_neighbors(g, node1, node2):

    node1_n = list(g[node1])

    node2_n = list(g[node2])

    common_n = np.intersect1d(node1_n, node2_n, assume_unique=True)

    return common_n



def adamic_adar_score(g, node1, node2):

    common_n = _common_neighbors(g, node1, node2)

    degrees = _common_degree(g, common_n)



    inv_log = np.divide(1., np.log(degrees + 1e-2))

    inv_log[inv_log < 0] = 0



    return np.sum(inv_log)



def _common_degree(g, common):

    N = common.shape[0]

    degrees = np.zeros(N, dtype=np.int)

    degrees[:] = [len(g[node]) for node in common]

    return degrees



def res_allocation(g, node1, node2):

    common_n = _common_neighbors(g, node1, node2)

    degrees = _common_degree(g, common_n)

    score = np.sum(np.divide(1., degrees + 1e-2))

    return score



@functools.lru_cache(maxsize=10)

def common_times(g, node1, node2):

    common = _common_neighbors(g, node1, node2)

    times = np.array(

        # get the friendship time for each of common friends

        [

            (g[node1][cf][1],

             g[node2][cf][1]) for cf in common

        ],

        dtype='float32'

    )

    return times



@functools.lru_cache(maxsize=10)

def common_forward_intensities(g, node1, node2):

    common = _common_neighbors(g, node1, node2)

    forward_inten = np.array(

        [

            (g[node1][cf][2],

             g[cf][node2][2]) for cf in common

        ],

        dtype='float32'

    )

    return forward_inten



METRICS = (

    common_neighbors_score,

    adamic_adar_score,

    res_allocation,

)



EDGE_FEATURES = (

    (common_times, functools.partial(np.mean, axis=0)),

    (common_times, functools.partial(np.min, axis=0)),

    (common_times, functools.partial(np.median, axis=0)),

    (common_forward_intensities, functools.partial(np.mean, axis=0)),

    (common_forward_intensities, functools.partial(np.max, axis=0)),

    (common_forward_intensities, functools.partial(np.median, axis=0)),

)



def get_all_features_for_pairs(g, pairs, edge_features=EDGE_FEATURES, metrics=METRICS, progress_bar=False):

    # Calculating edge features

    edge_features = np.full((len(pairs), len(EDGE_FEATURES), 2), -1, dtype='float32')

    if progress_bar:

        pairs_iter = enumerate(tqdm(pairs))

    else:

        pairs_iter = enumerate(pairs)

    for idx, row in pairs_iter:

        edge_features[idx] = compute_features(train_g, EDGE_FEATURES, row)

        

    edge_features = edge_features.reshape(-1, len(EDGE_FEATURES) * 2)

    

    if progress_bar:

        pairs_iter = enumerate(tqdm(pairs))

    else:

        pairs_iter = enumerate(pairs)

    metrics = np.full((len(pairs), len(METRICS)), -1, dtype='float32')

    for idx, row in pairs_iter:

        metrics[idx] = compute_metrics(g, METRICS, row)

    

    return edge_features, metrics



def assemble_all_features(pairs, edge_features, metrics, target):

    def get_uniqued_features(pairs, edge_features, metrics, target):

        num_pairs = pairs.shape[0]



        uniqued_pairs = np.full((num_pairs // 2, pairs.shape[1]), -1, dtype='int32')

        uniqued_metrics = np.full((num_pairs // 2, metrics.shape[1]), -1, dtype='float32')

        uniqued_target = np.full((num_pairs // 2), -1, dtype='int8')



        edge_features_num = edge_features.shape[1]



        # Uniqued edge features is 2 times wider than the original edge features,

        # because we can't just discard it, like metrics.

        # These features are directed, e.g. edge_features(a, b) != edge_features(b, a).

        uniqued_edge_features = np.full((num_pairs // 2, edge_features_num * 2), -1, dtype='float32')



        seen = {}

        idx = 0

        for pair, p_metric, p_edge_feature, p_target in zip(pairs, metrics, edge_features, target):

            sorted_pair = tuple(sorted(pair))

            if sorted_pair not in seen:

                uniqued_pairs[idx] = sorted_pair

                uniqued_metrics[idx] = p_metric

                uniqued_target[idx] = p_target

                uniqued_edge_features[idx][:edge_features_num] = p_edge_feature

                seen[sorted_pair] = idx

                idx += 1

            else:

                seen_idx = seen[sorted_pair]

                uniqued_edge_features[seen_idx][edge_features_num:] = p_edge_feature

        

        return uniqued_pairs, uniqued_edge_features, uniqued_metrics, uniqued_target

    

    def get_duplicate_columns(df):

        '''

        Get a list of duplicate columns.

        It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.

        :param df: Dataframe object

        :return: List of columns whose contents are duplicates.

        '''

        duplicateColumnNames = set()

        # Iterate over all the columns in dataframe

        for x in range(df.shape[1]):

            # Select column at xth index.

            col = df.iloc[:, x]

            # Iterate over all the columns in DataFrame from (x+1)th index till end

            for y in range(x + 1, df.shape[1]):

                # Select column at yth index.

                otherCol = df.iloc[:, y]

                # Check if two columns at x 7 y index are equal

                if col.equals(otherCol):

                    duplicateColumnNames.add(df.columns.values[y])



        return list(duplicateColumnNames)

    

    

    pairs, edge_features, metrics, target = get_uniqued_features(pairs, edge_features, metrics, target)

    gc.collect()

    

    # drop duplicate columns of edge_features (there always are some duplicate columns,

    # because some features of edges are the same in both directions)

    edge_features = pd.DataFrame(edge_features)

    if DROP_FIXED_DUPLICATE:

        duplicate_columns = [12, 13, 14, 15, 16, 17]

    else:

        duplicate_columns = get_duplicate_columns(edge_features.iloc[:1000, :])

    #     print(f'Found duplicate columns: {duplicate_columns}')

    edge_features = edge_features.drop(columns=duplicate_columns)

    

    

    # merge all features from pieces

    all_features = pd.DataFrame(pairs).rename(columns={0: 'id1', 1: 'id2'})

    all_features = pd.concat(

        (

            all_features,

            pd.DataFrame(metrics).rename(columns={0: 'CN', 1: 'AA', 2: 'RA'}),

            edge_features.add_suffix('_feat'),

            pd.Series(target, name='target', dtype='bool')

        ), axis=1)

    

    return all_features
train_g = nx.read_gpickle(CACHE_IN_FOLDER / 'train_graph.pkl')



test_pos_pairs = pd.read_csv(CACHE_IN_FOLDER / 'test_edges.csv', usecols=['id1', 'id2'])

print("Calculating all features for positive test pairs")

test_edge_features, test_metrics = get_all_features_for_pairs(train_g, test_pos_pairs.values, progress_bar=DEBUG)

print("Assembling all features")

test_pos = assemble_all_features(test_pos_pairs.values, test_edge_features, test_metrics, np.ones(test_pos_pairs.shape[0]))

print("Done")

del test_edge_features, test_metrics, train_g
positive_test_edges = collections.defaultdict(list)

for idx, row in tqdm(test_pos[['id1', 'id2']].iterrows()):

    positive_test_edges[row['id1']].append(idx)

    positive_test_edges[row['id2']].append(idx)

positive_test_edges = dict(positive_test_edges)



nodes_pos_edges = {}

for node, indices in tqdm(positive_test_edges.items(), mininterval=5):

    nodes_pos_edges[node] = test_pos.loc[indices, :]



nodes_neg_edges = {}

for idx, (node, n_pos_edges) in enumerate(tqdm(positive_test_edges.items(), mininterval=5)):

    candidates = all_features[((all_features['id1'] == node) | (all_features['id2'] == node)) & (all_features['target'] == False)]

    n_neg_edges = len(n_pos_edges) * NEG_EDGES_MULTIPLIER

    n_neg_edges = min(n_neg_edges, len(candidates))

    neg_edges = candidates.sample(n=n_neg_edges, random_state=RANDOM_SEED)

    nodes_neg_edges[node] = neg_edges

assert set(nodes_pos_edges.keys()) == set(nodes_neg_edges.keys())
indices = []

for idx, (node, edges) in enumerate(tqdm(nodes_neg_edges.items(), mininterval=5)):

    indices.extend(edges.index)

all_features.loc[indices, 'in_test'] = True
nodes_test_edges = {node: pd.concat([pos_edges, nodes_neg_edges[node]])

                    for node, pos_edges in nodes_pos_edges.items()}

assert set(nodes_test_edges.keys()) == set(nodes_pos_edges.keys()) == set(nodes_neg_edges.keys())

del nodes_pos_edges, nodes_neg_edges

gc.collect()
%%time

print("Saving")

with open(CACHE_OUT_FOLDER / 'test_set.pkl', 'wb') as f:

    pickle.dump(nodes_test_edges, f)

all_features.to_hdf(CACHE_OUT_FOLDER / 'all_features.h5', key='k', mode='w')

print("Done")