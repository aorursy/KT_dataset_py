#!pip -q install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl

#!pip -q install torchvision
## https://stackoverflow.com/questions/49853303/how-to-install-pydot-graphviz-on-google-colab?rq=1

#!pip -q install graphviz 

#!apt-get install graphviz -qq

#!pip -q install pydot
#!pip -q install "dask[complete]"
## https://medium.com/@iphoenix179/running-cuda-c-c-in-jupyter-or-how-to-run-nvcc-in-google-colab-663d33f53772

## https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal

#!apt update -qq;

#!wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb

#!mv cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64-deb cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb  

#!dpkg -i cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb

#!apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub

#!apt-get update -qq;

#!apt-get install cuda gcc-5 g++-5 -y -qq;

#!ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc;

#!ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++;

### !apt install cuda-9.2;
## http://alisonrowland.com/articles/installing-pycuda-via-pip

## https://codeyarns.com/2015/07/31/pip-install-error-with-pycuda/

#import os

#PATH = os.environ["PATH"]

#os.environ["PATH"] = "/usr/local/cuda-9.2/bin:/usr/local/cuda/bin:" + PATH

#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"

#os.environ["CUDA_ROOT"] = "/usr/local/cuda/"
#!pip -q install --ignore-installed pycuda
import numpy as np

import pandas as pd

from multiprocessing import Pool, Process

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import gc

gc.enable()
import multiprocessing as mp

mp.cpu_count()
def function(lst):

    arr = np.zeros_like(lst)

    for i in range(lst.shape[0]):

        for j in range(lst.shape[1]):

            arr[i][j] = lst[i][j] ** 2

    return arr



array = np.random.randint(1, 9, (2**10, 10000))

data = np.array_split(array, 2)
data[0].shape
%%time

with Pool(2) as p: # Only 2 processes will be called at a time, others will be in a queue.

    res = p.map(function, data)

    p.close()

    p.join()
%%time

processes = []

for i in range(2): # You can make as many processes as you like, they all will be working together until finished.

    p = Process(target=function, args=(data[i],))

    processes.append(p)

    p.start()

  

for p in processes: p.join()
from sklearn.datasets import make_regression

X, y = make_regression()



df = pd.DataFrame(X)

df.head()
dfs = [df.iloc[i*25:i*25+25, 0] for i in range(4)]

with Pool(4) as p:

    res = p.map(np.exp, dfs)

for i in range(4): df.iloc[i*25:i*25+25, 0] = res[i]

df.head()
def function(lst):

    arr = np.zeros_like(lst)

    for i in range(lst.shape[0]):

        arr[i] = lst[i] ** 2

    return arr
import time



def serial(n):

    times = []

    size = []

    for i in range(n):

        s = 10**(i+1)

        size.append(s)

        lst = np.random.randint(1, 7, (s,))

        st = time.time()

        res = function(lst)

        en = time.time()

        times.append(en-st)

    return times, size



def parallel(n):

    times = []

    size = []

    for i in range(n):

        s = 10**(i+1)

        size.append(s)

        lst = np.random.randint(1, 7, (s,))

        splitted = np.split(lst, 2)

        with Pool(2) as p:

            st = time.time()

            res = p.map(function, splitted)

            en = time.time()

        times.append(en-st)

    return times, size



def parallel2(n):

    times = []

    size = []

    for i in range(n):

        s = 10**(i+1)

        size.append(s)

        lst = np.random.randint(1, 7, (s,))

        splitted = np.split(lst, 2)

        processes = []

        for i in range(2):

            p = Process(target=function, args=(splitted[i],))

            processes.append(p)

        st = time.time()

        for p in processes: p.start()

        for p in processes: p.join()

        en = time.time()

        times.append(en-st)

    return times, size
t1, s1 = serial(7)
t2, s2 = parallel(7)
t3, s3 = parallel2(7)
plt.plot(s1, t1, "o-", label="Serial")

plt.plot(s2, t2, "o-", label="Pool")

plt.plot(s3, t3, "o-", label="Process")

plt.legend()

plt.xlabel("Number of elements:")

plt.ylabel("Time (sec):")

plt.show()

# Our task is not that complex, results here may vary unexpectedly.
from threading import Thread as trd

import queue

q = queue.Queue()
def function(lst):

    arr = np.zeros_like(lst)

    for i in range(lst.shape[0]):

        for j in range(lst.shape[0]):

            arr[i][j] = lst[i][j] * lst[i][j]

    return arr



array = np.random.randint(1, 10, (1000, 10000))

data = np.array_split(array, 2)
%%time

res = function(array)
%%time

# By using Queue this way you can get result of function without

# modifying your function.

t1 = trd(target=lambda q, args1: q.put(function(args1)), args=(q, data[0]))

t2 = trd(target=lambda q, args1: q.put(function(args1)), args=(q, data[1]))



t1.start()

t2.start()



t1.join()

t2.join()



res1 = q.get()

res2 = q.get()
q.empty()
from dask import delayed as delay



@delay

def add(x, y): return x+y

@delay

def sq(x): return x**2

@delay

def sum(x): 

    sum=0

    for i in range(len(x)): sum+=x[i]

    return sum
inputs = list(np.arange(1, 11))



res = [sq(n) for n in inputs]

res = [add(n, m) for n, m in zip(res[::2], res[1::2])]

res = sum(res)
res.visualize()
res.compute()
import torch.multiprocessing as mp_

mp = mp_.get_context('spawn')
a = torch.zeros((1000, 1000))

b = torch.zeros_like(a).cuda()
def func(arr):

    for i in range(arr.shape[0]):

        for j in range(arr.shape[1]):

            arr[i][j] += (i+j)

            arr[i][j] *= arr[i][j]

    return arr
%%time

res = func(a)
%%time

res = func(b)
import torch.multiprocessing as mp_

mp = mp_.get_context('spawn')
from sklearn.datasets import make_classification

from torch.utils.data import DataLoader, TensorDataset

X, y = make_classification(n_samples=100000, )



dataset = TensorDataset(torch.FloatTensor(X), torch.DoubleTensor(y))

data_loader = DataLoader(dataset, batch_size=8)
n_in = 20; n_out = 1        

        

model = nn.Sequential(nn.Linear(n_in, 15),

                      nn.ReLU(),

                      nn.Linear(15, 10),

                      nn.ReLU(),

                      nn.Linear(10, 5),

                      nn.ReLU(),

                      nn.Linear(5, n_out),

                      nn.Sigmoid())



model.share_memory() # Required for 'fork' method to work
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)

loss_fun = torch.nn.BCELoss()
import torch.nn as nn

def train(model):

    for data, labels in data_loader:

        optimizer.zero_grad()

        loss_fn(model(data), labels).backward()

        optimizer.step()  # This will update the shared parameters
processes = []

for i in range(4): # No. of processes

    p = mp.Process(target=train, args=(model,))

    p.start()

    processes.append(p)

for p in processes: p.join()
sum=0

for data, labels in data_loader:

    with torch.no_grad():

        res = model(data)

        res[res>=0.7] = 1

        res[res<0.7] = 0

        sum += (res.numpy()!=labels.float().numpy()).diagonal().sum()

    

sum/100000