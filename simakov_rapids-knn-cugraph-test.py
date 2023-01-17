import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf

import cugraph

from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

import numpy as np

import igraph

import time
for j in [1e4, 1e5]:

    n_sample = int(j)

    dim = 5



    n_neighbors = 2

    res = []

    t0 = time.time()

    for i in range(100):

        np.random.seed(n_sample + i)

        X = np.random.rand(n_sample, dim)

        device_data = cudf.DataFrame.from_gpu_matrix(X)

        

        knn_cuml = cuNearestNeighbors(n_neighbors)

        knn_cuml.fit(device_data)

        D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

        

        g = igraph.Graph(directed = True)

        g.add_vertices(range(n_sample))

        g.add_edges(I_cuml.to_pandas().values)

        g2 = g.as_undirected(mode = 'collapse')

        r = g2.clusters()

        

        res.append(len(r) / n_sample * 100)

    print(f'n: {int(j)}, mean coef: {np.round(np.mean(res), 3)}, time: {np.round(time.time() - t0, 3)} s')
for j in [1e4, 1e5]:

    n_sample = int(j)

    dim = 5



    n_query = n_sample

    n_neighbors = 2

    random_state = 0

    res = []

    t0 = time.time()

    for i in range(100):

        np.random.seed(n_sample + i)

        X = np.random.rand(n_sample, dim)

        device_data = cudf.DataFrame.from_gpu_matrix(X)

        

        knn_cuml = cuNearestNeighbors(n_neighbors)

        knn_cuml.fit(device_data)

        D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

        

        G = cugraph.Graph()

        I_cuml = cugraph.structure.symmetrize_df(I_cuml, 0, 1)

        G.from_cudf_edgelist(I_cuml, 0, 1)

        

        res.append(len(cugraph.weakly_connected_components(G).labels.unique()) / n_sample * 100)

    print(f'n: {int(j)}, mean coef: {np.round(np.mean(res), 3)}, time: {np.round(time.time() - t0, 3)} s')