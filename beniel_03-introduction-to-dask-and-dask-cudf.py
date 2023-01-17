!nvidia-smi
!nvcc --version
import sys
!rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
!rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import cudf
import os
try:
    import graphviz
except ModuleNotFoundError:
    os.system('conda install -c conda-forge graphviz -y')
    os.system('conda install -c conda-forge python-graphviz -y')
import dask; print('Dask Version:', dask.__version__)
from dask.distributed import Client, LocalCluster


# create a local cluster with 4 workers
n_workers = 4
cluster = LocalCluster(n_workers=n_workers)
client = Client(cluster)
# show current Dask status
client
def add_5_to_x(x):
    return x + 5
from dask import delayed


addition_operations = [delayed(add_5_to_x)(i) for i in range(n_workers)]
addition_operations
total = delayed(sum)(addition_operations)
total
total.visualize()
from dask.distributed import wait
import time


addition_futures = client.compute(addition_operations, optimize_graph=False, fifo_timeout="0ms")
total_future = client.compute(total, optimize_graph=False, fifo_timeout="0ms")
wait(total_future)  # this will give Dask time to execute the work
addition_futures
print(total_future)
print(type(total_future))
addition_results = [future.result() for future in addition_futures]
print('Addition Results:', addition_results)
addition_results = client.gather(addition_futures)
total_result = client.gather(total_future)
print('Addition Results:', addition_results)
print('Total Result:', total_result)
def sleep_1():
    time.sleep(1)
    return 'Success!'
%%time

for _ in range(n_workers):
    sleep_1()
%%time

# define delayed execution graph
sleep_operations = [delayed(sleep_1)() for _ in range(n_workers)]

# use client to perform computations using execution graph
sleep_futures = client.compute(sleep_operations, optimize_graph=False, fifo_timeout="0ms")

# collect and print results
sleep_results = client.gather(sleep_futures)
print(sleep_results)
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


# create a local CUDA cluster
cluster = LocalCUDACluster()
client = Client(cluster)
client
import cudf; print('cuDF Version:', cudf.__version__)
import numpy as np; print('NumPy Version:', np.__version__)


def load_data(n_rows):
    df = cudf.DataFrame()
    random_state = np.random.RandomState(43210)
    df['key'] = random_state.binomial(n=1, p=0.5, size=(n_rows,))
    df['value'] = random_state.normal(size=(n_rows,))
    return df
def head(dataframe):
    return dataframe.head()
# define the number of workers
n_workers = 1  # feel free to change this depending on how many GPUs you have

# define the number of rows each dataframe will have
n_rows = 125000000  # we'll use 125 million rows in each dataframe
from dask.delayed import delayed


# create each dataframe using a delayed operation
dfs = [delayed(load_data)(n_rows) for i in range(n_workers)]
dfs
head_dfs = [delayed(head)(df) for df in dfs]
head_dfs
from dask.distributed import wait


# use the client to compute - this means create each dataframe and take the head
futures = client.compute(head_dfs)
wait(futures)  # this will give Dask time to execute the work before moving to any subsequently defined operations
futures
# collect the results
results = client.gather(futures)
results
# let's inspect the head of the first dataframe
print(results[0])
def length(dataframe):
    return dataframe.shape[0]
lengths = [delayed(length)(df) for df in dfs]
lengths
total_number_of_rows = delayed(sum)(lengths)
total_number_of_rows.visualize()
# use the client to compute the result and wait for it to finish
future = client.compute(total_number_of_rows)
wait(future)
future
# collect result
result = client.gather(future)
result
def groupby(dataframe):
    return dataframe.groupby('key')['value'].mean()
groupbys = [delayed(groupby)(df) for df in dfs]
# use the client to compute the result and wait for it to finish
groupby_dfs = client.compute(groupbys)
wait(groupby_dfs)
groupby_dfs
results = client.gather(groupby_dfs)
results
for i, result in enumerate(results):
    print('cuDF DataFrame:', i)
    print(result)
import dask_cudf; print('Dask cuDF Version:', dask_cudf.__version__)


# create a distributed cuDF DataFrame using Dask
distributed_df = dask_cudf.from_delayed(dfs)
print('Type:', type(distributed_df))
distributed_df
result = distributed_df.groupby('key')['value'].mean().compute()
result
print(result)