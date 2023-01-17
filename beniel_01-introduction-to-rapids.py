# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!nvidia-smi
# Check Python Version

!python --version
# Check CUDA/cuDNN Version

!nvcc -V && which nvcc
import sys

!rsync -ah --progress ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!rsync -ah --progress /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
!conda activate rapids

!conda install -c rapidsai dask-cudf




import numpy as np; print('NumPy Version:', np.__version__)





# create the relationship: y = 2.0 * x + 1.0

n_rows = 40000  # let's use 100 thousand data points

w = 2.0

x = np.random.normal(loc=0, scale=1, size=(n_rows,))

b = 1.0

y = w * x + b



# add a bit of noise

noise = np.random.normal(loc=0, scale=2, size=(n_rows,))

y_noisy = y + noise



try:

    import matplotlib

except ModuleNotFoundError:

    os.system('conda install -y matplotlib')

from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt





%matplotlib inline
plt.scatter(x, y_noisy, label='empirical data points')

plt.plot(x, y, color='black', label='true relationship')

plt.legend()
import sklearn; print('Scikit-Learn Version:', sklearn.__version__)

from sklearn.linear_model import LinearRegression





# instantiate and fit model

linear_regression = LinearRegression()
%%time



linear_regression.fit(np.expand_dims(x, 1), y)
# create new data and perform inference

inputs = np.linspace(start=-5, stop=5, num=1000)

outputs = linear_regression.predict(np.expand_dims(inputs, 1))
plt.scatter(x, y_noisy, label='empirical data points')

plt.plot(x, y, color='black', label='true relationship')

plt.plot(inputs, outputs, color='red', label='predicted relationship (cpu)')

plt.legend()
import cudf; print('cuDF Version:', cudf.__version__)
# create a cuDF DataFrame

df = cudf.DataFrame({'x': x, 'y': y_noisy})

print(df.head())
import cuml; print('cuML Version:', cuml.__version__)

from cuml.linear_model import LinearRegression as LinearRegression_GPU





# instantiate and fit model

linear_regression_gpu = LinearRegression_GPU()
%%time



linear_regression_gpu.fit(df[['x']], df['y'])
# create new data and perform inference

new_data_df = cudf.DataFrame({'inputs': inputs})

outputs_gpu = linear_regression_gpu.predict(new_data_df[['inputs']])
plt.scatter(x, y_noisy, label='empirical data points')

plt.plot(x, y, color='black', label='true relationship')

plt.plot(inputs, outputs, color='red', label='predicted relationship (cpu)')

plt.plot(inputs, outputs_gpu.to_array(), color='green', label='predicted relationship (gpu)')

plt.legend()
import dask; print('Dask Version:', dask.__version__)

from dask.distributed import Client, LocalCluster





# create a local cluster with 4 workers

n_workers = 4

cluster = LocalCluster(n_workers=n_workers)

client = Client(cluster)
# show current Dask status

client
import time





def sleep_1():

    time.sleep(1)

    return 'Success!'
%%time



for _ in range(n_workers):

    sleep_1()
from dask.delayed import delayed
%%time



# define delayed execution graph

sleep_operations = [delayed(sleep_1)() for _ in range(n_workers)]



# use client to perform computations using execution graph

sleep_futures = client.compute(sleep_operations, optimize_graph=False, fifo_timeout="0ms")



# collect and print results

sleep_results = client.gather(sleep_futures)

print(sleep_results)
import dask; print('Dask Version:', dask.__version__)

from dask.distributed import Client

# import dask_cuda; print('Dask CUDA Version:', dask_cuda.__version__)

from dask_cuda import LocalCUDACluster





# create a local CUDA cluster

cluster = LocalCUDACluster()

client = Client(cluster)
client
import dask_cudf; print('Dask cuDF Version:', dask_cudf.__version__)





# identify number of workers

workers = client.has_what().keys()

n_workers = len(workers)



# create a cuDF DataFrame with two columns named "key" and "value"

df = cudf.DataFrame()

n_rows = 100000000  # let's process 100 million rows in a distributed parallel fashion

df['key'] = np.random.binomial(1, 0.2, size=(n_rows))

df['value'] = np.random.normal(size=(n_rows))



# create a distributed cuDF DataFrame using Dask

distributed_df = dask_cudf.from_cudf(df, npartitions=n_workers)
n_workers
# inspect our distributed cuDF DataFrame using Dask

print('-' * 15)

print('Type of our Dask cuDF DataFrame:', type(distributed_df))

print('-' * 15)

print(distributed_df.head())
aggregation = distributed_df['value'].sum()

print(aggregation.compute())