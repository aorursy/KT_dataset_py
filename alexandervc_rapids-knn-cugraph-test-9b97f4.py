dim = 50000
data_type = 'Torus Uniform'# 'Sphere Uniform'
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
cuNearestNeighbords(1)
for j in [1e2,1e3]:#,2e3,4e3,5e3]:#,1e5,1e6]:
    n_sample = int(j)
    

    n_neighbors = 2
    res = []
    t0 = time.time()
    for i in range(1):
        np.random.seed(n_sample + i)
        #Dorm'
        if 0:
            X = np.random.rand(n_sample, dim) # uniform
            X = np.random.randn(n_sample, dim) # Gaussian
        if data_type == 'Torus Uniform':
            X0 = np.random.rand(n_sample, dim)
            X = np.concatenate( (np.sin(2*np.pi* X0), np.cos(2*np.pi*X0 ) ) , axis = 1) 
        elif data_type == 'Sphere Uniform': 
            X = np.random.randn(n_sample, dim)
            norms = np.sqrt( np.sum(X*X,1) )
            X = X/norms.reshape(len(norms),-1) # Sphere 
        device_data = cudf.DataFrame.from_gpu_matrix(X)
        
        knn_cuml = cuNearestNeighbords(n_neighbors)
        knn_cuml.fit(device_data)
        D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)
        
        g = igraph.Graph(directed = True)
        g.add_vertices(range(n_sample))
        g.add_edges(I_cuml.to_pandas().values)
        g2 = g.as_undirected(mode = 'collapse')
        r = g2.clusters()
        
        res.append(len(r) / n_sample * 100)
    print(f'n: {int(j)}, mean coef: {np.round(np.mean(res), 3)}, time: {np.round(time.time() - t0, 3)} s')
import pandas as pd
df_stat = pd.DataFrame()


c = 0
timer_for_print = time.time()
t00 = time.time()
time_interval_for_print = 1000 # seconds

for n_sample in [1e3,2e3,5e3]:
    n_trials = 100
    #if n_sample <= 1e3:
    #    n_trials = 1000
    if n_sample > 1e6:
        n_trials = 10
    n_sample = int(n_sample)

    n_neighbors = 2
    res = []
    res2 = []
    res3 = []
    for i in range(n_trials):
        t0 = time.time()
        c+=1
        df_stat.loc[c,'Dim'] = dim
        df_stat.loc[c,'sample size'] = n_sample
        
        np.random.seed(n_sample + i)
        #data_type = 'Sphere Uniform' # Define above
        df_stat.loc[c,'Data'] = data_type
        if 0:
            X = np.random.rand(n_sample, dim) # uniform
            X = np.random.randn(n_sample, dim) # Gaussian
        if data_type == 'Torus Uniform':
            X0 = np.random.rand(n_sample, dim)
            X = np.concatenate( (np.sin(2*np.pi* X0), np.cos(2*np.pi*X0 ) ) , axis = 1) 
        elif data_type == 'Sphere Uniform': 
            X = np.random.randn(n_sample, dim)
            norms = np.sqrt( np.sum(X*X,1) )
            X = X/norms.reshape(len(norms),-1) # Sphere 
        
        
        device_data = cudf.DataFrame.from_gpu_matrix(X)
        
        knn_cuml = cuNearestNeighbors(n_neighbors)
        knn_cuml.fit(device_data)
        D_cuml, I_cuml = knn_cuml.kneighbors(device_data, n_neighbors)
        
        t_part1 = np.round(time.time() - t0, 3)
        
        g = igraph.Graph(directed = True)
        g.add_vertices(range(n_sample))
        g.add_edges(I_cuml.to_pandas().values)
        r = g.clusters(mode = 'WEAK')
        
        res.append(len(r) / n_sample * 100)
        df_stat.loc[c,'%Coeff'] = len(r) / n_sample * 100
        
        comp_size = [len(tmp) for tmp in r ]
        res2.append(np.max(comp_size))
        res3.append(np.mean(comp_size))
        df_stat.loc[c,'Max comp size'] = np.max(comp_size)
        df_stat.loc[c,'Mean comp size'] = np.mean(comp_size)
        df_stat.loc[c,'Median comp size'] = np.median(comp_size)
        df_stat.loc[c,'Seconds passed'] = time.time() - t0
        df_stat.loc[c,'n_trials'] = n_trials
        #print(np.max(comp_size))
        if  timer_for_print - time.time() >= time_interval_for_print :
            time_interval_for_print = time.time()
            print('c',c,'i',i,'dim', dim, f'n: {int(n_sample)}, mean coef: {np.round(np.mean(res), 3)}, std coef: {np.round(np.std(res), 6)}', 
              #f' mean max_comp_size: {np.round(np.mean(res2), 1)}, std max_comp_size: {np.round(np.std(res2), 3)}',
              #f' mean mean_comp_size: {np.round(np.mean(res3), 1)}, std mean_comp_size: {np.round(np.std(res3), 3)}',
              f'time knn graph: {t_part1}, full time: {np.round(time.time() - t0, 3)} s')
        #print(df_stat.tail(1))
    print('Finsihed  for n_sample', n_sample)
    print('c',c,'i',i,'dim', dim, f'n: {int(n_sample)}, mean coef: {np.round(np.mean(res), 3)}, std coef: {np.round(np.std(res), 6)}', 
      f' mean max_comp_size: {np.round(np.mean(res2), 1)}, std max_comp_size: {np.round(np.std(res2), 3)}',
      f' mean mean_comp_size: {np.round(np.mean(res3), 1)}, std mean_comp_size: {np.round(np.std(res3), 3)}',
      f'time knn graph: {t_part1}, full time: {np.round(time.time() - t0, 3)} s')
    print(res)
    print()
    
print(); print()
print('STD groupby sample size')
print(df_stat.groupby('sample size').std() )
print(); print()
print('MEAN groupby sample size')
print(df_stat.groupby('sample size').mean() )
print('Total time:', time.time() - t00, 'seconds passed')
df_stat.groupby('sample size').mean() 
#df_stat

df_stat.groupby('sample size').mean() 


df_stat.groupby('sample size').std() 






dddddddddddddddddddddddddddddddd