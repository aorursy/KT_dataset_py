# Use rapids - https://www.kaggle.com/cdeotte/rapids - some GPU acceleration for many algorithms

# Based on Dmitry Simakov notebook: https://www.kaggle.com/simakov/rapids-knn-cugraph-test



# That cell may run 3-5 minutes. (And sometimes may hang on - if so - restart notebook and run again )

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

import matplotlib.pyplot as plt 
# Make Loglog plot of the degrees - we will see almost a line for D=100,1000

# That means that count = a* (degree)^b for some a,b - power law - so graph is "scale-free" network https://en.wikipedia.org/wiki/Scale-free_network





n_sample = 10**5 # 10**6 and dim 1000 will crash due to memory (Kaggle kernel)

n_neighbors = 2



plt.style.use('ggplot')



for dim in [2,10,100,1000]: #= 1000

    print('dimension = ',dim)

    t0=time.time()



    X = np.random.rand(n_sample, dim)

    device_data = cudf.DataFrame.from_gpu_matrix(X)

    knn_cuml = cuNearestNeighbors(n_neighbors)

    knn_cuml.fit(device_data)

    D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

    print(time.time()-t0,'secs passed. Knn done')



    g = igraph.Graph(directed = True)

    g.add_vertices(range(n_sample))

    g.add_edges(I_cuml.to_pandas().values)

    print(time.time()-t0,'secs passed. Graph done')



    g2 = g.as_undirected(mode = 'collapse')

    list_degs = g2.degree()

    bins = range(np.max(list_degs))

    h = np.histogram(list_degs, bins)

    #print(h)

    label = 'Dim '+str(dim)

    plt.loglog( h[1][:-1], h[0] , '*-', label = label)





plt.title('Degree distribution (UNoriented) ')

plt.xlabel('degree')

plt.ylabel('Count')

plt.legend()

plt.show()

# Same as above, but for IN-degrees, result is the same as above, but may be a little worse 

#    # Make Loglog plot of the degrees - we will see almost a line for D=100,1000

#    # That means that count = a* (degree)^b for some a,b - power law - so graph is "scale-free" network https://en.wikipedia.org/wiki/Scale-free_network





n_sample = 10**5

n_neighbors = 2



plt.style.use('ggplot')



for dim in [2,10,100,1000]: #= 1000

    print('dimension = ',dim)

    t0=time.time()



    X = np.random.rand(n_sample, dim)

    device_data = cudf.DataFrame.from_gpu_matrix(X)

    knn_cuml = cuNearestNeighbors(n_neighbors)

    knn_cuml.fit(device_data)

    D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

    print(time.time()-t0,'secs passed. Knn done')



    g = igraph.Graph(directed = True)

    g.add_vertices(range(n_sample))

    g.add_edges(I_cuml.to_pandas().values)

    print(time.time()-t0,'secs passed. Graph done')



    list_degs = g.indegree()

    bins = range(np.max(list_degs))

    h = np.histogram(list_degs, bins)

    #print(h)

    label = 'Dim '+str(dim)

    plt.loglog( h[1][:-1], h[0] , '*-', label = label)





plt.title('IN Degree distribution')

plt.xlabel('degree')

plt.ylabel('Count')

plt.legend()

plt.show()

# Gauss distribution 



# Make Loglog plot of the degrees - we will see almost a line for D=100,1000

# That means that count = a* (degree)^b for some a,b - power law - so graph is "scale-free" network https://en.wikipedia.org/wiki/Scale-free_network





n_sample = 10**5 # 10**6 and dim 1000 will crash due to memory (Kaggle kernel)

n_neighbors = 2



plt.style.use('ggplot')



for dim in [50, 1000]: #= 1000

    print('dimension = ',dim)

    fig = plt.figure(figsize = (15,8) )

    for mode in ['Uniform','Gauss']:

        t0=time.time()



        if mode == 'Uniform':

            X = np.random.rand(n_sample, dim)

        else:

            X = np.random.randn(n_sample, dim)

            

        device_data = cudf.DataFrame.from_gpu_matrix(X)

        knn_cuml = cuNearestNeighbors(n_neighbors)

        knn_cuml.fit(device_data)

        D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

        print(time.time()-t0,'secs passed. Knn done')



        g = igraph.Graph(directed = True)

        g.add_vertices(range(n_sample))

        g.add_edges(I_cuml.to_pandas().values)

        print(time.time()-t0,'secs passed. Graph done')



        list_degs = g.indegree()

        bins = range(np.max(list_degs))

        h = np.histogram(list_degs, bins)

        #print(h)

        label = 'Dim '+str(dim) + ' '+ mode + ' IN-degs'

        plt.loglog( h[1][:-1], h[0] , '*-', label = label)



        g2 = g.as_undirected(mode = 'collapse')

        list_degs = g2.degree()

        bins = range(np.max(list_degs))

        h = np.histogram(list_degs, bins)

        #print(h)

        label = 'Dim '+str(dim) + ' '+ mode + ' degs'

        plt.loglog( h[1][:-1], h[0] , '*-', label = label)



    



    plt.title('Degree distribution')

    plt.xlabel('degree')

    plt.ylabel('Count')

    plt.legend()

    plt.show()
