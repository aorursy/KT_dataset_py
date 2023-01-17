# Use rapids - https://www.kaggle.com/cdeotte/rapids - some GPU acceleration for many algorithms

# 

# Current notebook  is based on Dmitry Simakov notebook: https://www.kaggle.com/simakov/rapids-knn-cugraph-test



# That cell may run 1-3-5 minutes. (And sometimes may hang on - if so - restart notebook and run again )

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

#import igraph

import time
import time

from sklearn.neighbors import NearestNeighbors

import numpy as np

import pandas as pd





#dim = 10

c = 0 

#method = 'kd_tree'

df_stat = pd.DataFrame()



for dim in [5,10,20]:

  for n_sample in [1e5,2e5]:

    n_sample = int(n_sample)



    n_neighbors = 2

    res = []

    for i in range(1): # Repeat test several times

        np.random.seed(n_sample + i)

        X = np.random.rand(n_sample, dim)

        

        for method in ['brute','kd_tree','ball_tree','GPU']:

            c += 1



            df_stat.loc[c, 'Method'] = method

            df_stat.loc[c, 'Dim'] = dim

            df_stat.loc[c, 'N_sample'] = n_sample

            t0 = time.time()

            t00 = t0

            if method == 'GPU':

              device_data = cudf.DataFrame.from_gpu_matrix(X)

              knn_cuml = cuNearestNeighbors(n_neighbors)

              knn_cuml.fit(device_data)

              D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

              indices = I_cuml.to_pandas().values

            else:

              nbrs = NearestNeighbors(n_neighbors=2, algorithm=method  ).fit(X) # 'ball_tree'

              distances, indices = nbrs.kneighbors(X)

            df_stat.loc[c, 'Time NN'] = time.time()-t0

            

            if method == 'brute':

              indices_save = indices.copy()

            difr = indices_save - indices

            df_stat.loc[c, 'Coincide with Brute'] = (np.sum( np.abs(difr)) == 0) 

            print('c',c,'Dim',dim,'Finished.',method, np.round(time.time()-t0,2),'secs passed')



df_stat
import time

from sklearn.neighbors import NearestNeighbors

import numpy as np

import pandas as pd





#dim = 10

c = 0 

#method = 'kd_tree'

df_stat = pd.DataFrame()



for dim in [5,10,20,100, 200]:

  for n_sample in [1e5,5e5,1e6]:

    

    n_sample = int(n_sample)



    n_neighbors = 2

    res = []

    for i in range(1): # Repeat test several times

        np.random.seed(n_sample + i)

        X = np.random.rand(n_sample, dim)

        

        for method in ['GPU']:

            c += 1



            df_stat.loc[c, 'Method'] = method

            df_stat.loc[c, 'Dim'] = dim

            df_stat.loc[c, 'N_sample'] = n_sample

            t0 = time.time()

            t00 = t0

            if method == 'GPU':

              device_data = cudf.DataFrame.from_gpu_matrix(X)

              knn_cuml = cuNearestNeighbors(n_neighbors)

              knn_cuml.fit(device_data)

              D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)

              indices = I_cuml.to_pandas().values

            else:

              nbrs = NearestNeighbors(n_neighbors=2, algorithm=method  ).fit(X) # 'ball_tree'

              distances, indices = nbrs.kneighbors(X)

            df_stat.loc[c, 'Time NN'] = time.time()-t0

            print(df_stat.tail(1))

            



df_stat