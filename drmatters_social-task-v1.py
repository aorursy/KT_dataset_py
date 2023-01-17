import pandas as pd

import numpy as np

import random

import pathlib

import collections

import gc

import functools

import networkx as nx

import itertools

import operator

import os



from tqdm import tqdm

import seaborn as sns

tqdm.pandas()
DEBUG = False



USE_CLEANED_CACHE = True

USE_PAIRS_CACHE = True

USE_TARGET_CACHE = True

USE_METRICS_CACHE = True

USE_EDGE_FEATURES_CACHE = True



RANDOM_SEED = 42

CACHE_OUT_FOLDER = pathlib.Path('../working/cache')

CACHE_IN_FOLDER = pathlib.Path('../input/social-task-v1/cache')



random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)
if not os.path.exists(CACHE_OUT_FOLDER):

    os.makedirs(CACHE_OUT_FOLDER)
if not USE_CLEANED_CACHE:

    path = pathlib.Path('../input/socialtaskfriends/friends_dataset.csv')

    df = pd.read_csv(path, names=['id1', 'id2', 'time', 'inten'])

    print(df[['time', 'inten']].describe())

    df[:2]
CLEANED_FILENAME = 'cleaned.h5'

if USE_CLEANED_CACHE:

    print(f"Trying to read from {str(CACHE_IN_FOLDER / CLEANED_FILENAME)}")

    cleaned = pd.read_hdf(CACHE_IN_FOLDER / CLEANED_FILENAME)

    

else:

    # сохраняем информацию о дублирующихся ребрах: (id1, id2) -> List[int]

    data = collections.defaultdict(list)

    def find_intersections(row):

        id1 = row['id1']

        id2 = row['id2']

        higher = max(id1, id2)

        lower = min(id1, id2)



        data[(higher, lower)].append(row.name)



    df[['id1', 'id2']].progress_apply(find_intersections, axis=1)

    

    # Взглянем на данные

    lens = list(map(len, data.values()))

    srs = pd.Series(lens, dtype='int8')

    srs.name = "Number of edges"

    print("Description:\n", srs.describe())

    print("\nUnique values:\n", srs.value_counts())

    

    # удалим ребра, которые идут только в одном направлении, или дублируются

    outliers = [val for val in data.values() if len(val) != 2]

    outliers = np.concatenate(outliers).astype('int32')

    cleaned = df.drop(outliers, axis=0).reset_index(drop=True)



print(f"Saving")

cleaned.to_hdf(CACHE_OUT_FOLDER / CLEANED_FILENAME, key='k', mode='w')

print(f"Saved to {str(CACHE_OUT_FOLDER / CLEANED_FILENAME)}")
top_percentile = np.percentile(cleaned['time'], 10)

print('10% percentile:', top_percentile)

train_edges = cleaned[~(cleaned['time'] < top_percentile)]
train_g = nx.from_pandas_edgelist(train_edges.rename(columns={'time': 1, 'inten': 2}),

                                  source='id1', target='id2', edge_attr=[1, 2],

                                  create_using=nx.DiGraph)

print(f'Nodes: {len(train_g.nodes)}')

print(f'Edges: {len(train_g.edges)}')



print('Saving graph')

with open(CACHE_OUT_FOLDER / 'train_graph.pkl', 'wb') as f:

    nx.write_gpickle(train_g, f)

print('Saved')
full_g = nx.from_pandas_edgelist(cleaned.rename(columns={'time': 1, 'inten': 2}),

                                source='id1', target='id2', edge_attr=[1, 2],

                                create_using=nx.DiGraph)

print(f'Nodes: {len(full_g.nodes)}')

print(f'Edges: {len(full_g.edges)}')





print('Saving graph')

with open(CACHE_OUT_FOLDER / 'full_graph.pkl', 'wb') as f:

    nx.write_gpickle(full_g, f)

print('Saved')
missed_nodes = set(full_g.nodes) - set(train_g.nodes)

len(missed_nodes)
test_edges = cleaned[(cleaned['time'] < top_percentile)]

disappeared_edges_mask = test_edges.apply(

    lambda row: row['id1'] in missed_nodes or row['id2'] in missed_nodes,

    axis=1

)



predictable_test_edges = test_edges[~disappeared_edges_mask].loc[:, ['id1', 'id2']]

print('Total edges disappeared from test: ', len(test_edges[disappeared_edges_mask]))

print('Test edges retained (%): ', len(predictable_test_edges) * 100 / len(test_edges))
@functools.lru_cache()

def _common_neighbors(g, node1, node2):

    node1_n = list(g[node1])

    node2_n = list(g[node2])

    common_n = np.intersect1d(node1_n, node2_n, assume_unique=True)

    return common_n



has_common_friends = predictable_test_edges.progress_apply(

    lambda row: len(_common_neighbors(train_g, row['id1'], row['id2'])),

    axis=1

)

has_common_friends = has_common_friends != 0

predictable_test_edges = predictable_test_edges[has_common_friends]





# сохраняем для дальнейшего использования при тестах моделей

print(f"Saving")

predictable_test_edges.to_csv(CACHE_OUT_FOLDER / 'test_edges.csv', index=False)

print(f"Saved to {str(CACHE_OUT_FOLDER / 'test_edges.csv')}")
if USE_PAIRS_CACHE:

    print('Trying to read adjancencies')

    with open(CACHE_IN_FOLDER / 'adjs.npy', 'rb') as f:

        pairs_np = np.load(f)

else:

    print('Calculating adjancencies')

    all_adjs = {}

    for idx, node in enumerate(tqdm(sorted(train_g.nodes))):

        adjs = train_g[node]



        l2_adjs = set()

        for adj_node in adjs:

            l2_adjs = l2_adjs | set(train_g[adj_node])

        l2_adjs = l2_adjs - set([node])



        all_adjs[node] = sorted(l2_adjs)

    

    print('Calculating pairs')

    pairs = []

    for idx, (start, finish) in enumerate(tqdm(all_adjs.items())):

        first_col = np.full((len(finish)), start, dtype='int32')

        second_col = np.asarray(finish, dtype='int32')

        pairs.append(np.column_stack((first_col, second_col)))

    pairs_np = np.concatenate(pairs, axis=0)

    



print('Saving adjacency pairs')

with open(CACHE_OUT_FOLDER / 'adjs.npy', 'wb') as f:

    np.save(f, pairs_np)

print('Saved')
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



METRICS = [

    common_neighbors_score,

    adamic_adar_score,

    res_allocation,

]



EDGE_FEATURES = [

    (common_times, functools.partial(np.mean, axis=0)),

    (common_times, functools.partial(np.min, axis=0)),

    (common_times, functools.partial(np.median, axis=0)),

    (common_forward_intensities, functools.partial(np.mean, axis=0)),

    (common_forward_intensities, functools.partial(np.max, axis=0)),

    (common_forward_intensities, functools.partial(np.median, axis=0)),

]
edge_features = np.full((len(pairs_np), len(EDGE_FEATURES), 2), -1, dtype='float32')



if USE_EDGE_FEATURES_CACHE:

    print("Reading edge features")

    with open(CACHE_IN_FOLDER / 'edge_features.npy', 'rb') as f:

        edge_features = np.load(f)

else:

    for idx, row in enumerate(tqdm(pairs_np, maxinterval=120, mininterval=60)):

        edge_features[idx] = compute_features(train_g, EDGE_FEATURES, row)

    edge_features = edge_features.reshape(-1, len(EDGE_FEATURES) * 2)



print("Saving edge features")

with open(CACHE_OUT_FOLDER / 'edge_features.npy', 'wb') as f:

    np.save(f, edge_features)

print("Saved")
if USE_METRICS_CACHE:

    print("Reading metrics")

    with open(pathlib.Path(CACHE_IN_FOLDER / 'metrics.npy'), 'rb') as f:

        metrics = np.load(f)

else:

    metrics = np.full((len(pairs_np), len(METRICS)), -1, dtype='float32')

    for idx, row in enumerate(tqdm(pairs_np, mininterval=30, maxinterval=60)):

        metrics[idx] = compute_metrics(train_g, METRICS, row)



# save to cache anyway, to keep it persistent between Kaggle Kernel relaunch

print("Saving metrics")

with open(CACHE_OUT_FOLDER / 'metrics.npy', 'wb') as f:

    np.save(f, metrics)

print("Saved")
def are_friends(g, node1, node2):

    return node2 in g[node1]



if USE_TARGET_CACHE:

    print("Reading target")

    with open(CACHE_IN_FOLDER / 'target.npy', 'rb') as f:

        target = np.load(f)

else:

    target = np.array([are_friends(train_g, row[0], row[1])

                       for row in tqdm(pairs_np, mininterval=1, maxinterval=60)], dtype='bool')



print("Saving target")

with open(CACHE_OUT_FOLDER / 'target.npy', 'wb') as f:

    np.save(f, target)

print("Saved")