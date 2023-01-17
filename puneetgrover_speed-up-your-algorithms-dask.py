#!pip -q install "dask[complete]"
#!pip -q install "dask-ml[complete]"
#!pip -q install --upgrade --ignore-installed numpy pandas scipy sklearn
## https://stackoverflow.com/questions/49853303/how-to-install-pydot-graphviz-on-google-colab?rq=1
#!pip -q install graphviz 
#!apt-get install graphviz -qq
#!pip -q install pydot
#!pip -q install bokeh
import numpy as np
import pandas as pd
import dask.array as da
import graphviz
import matplotlib.pyplot as plt
arr = np.random.randint(1, 1000, (1000, 1000))
darr = da.from_array(arr, chunks=(250, 250))
darr
darr.visualize(color="order", size="9,10!")
darr.chunks, darr.chunksize, darr.npartitions
res = darr.sum(axis=0)
res.visualize(rankdir="LR", size="3,20!") # Graph of methods we applied
# If we have a graph structure with many independent nodes per level in our implementation, Dask will be able to 
# parallelize it and we will get speedup, if our problem is sufficiently large.
res.compute().shape
def numpy_mean(size=(10, 10)):
  arr = np.random.random(size=size)
  return arr.mean()

def dask_mean(size=(10, 10)):
  if size[0] > 10000: chunks = (1000, 1000)
  else: chunks = (int(size[0]/10), int(size[1]/10))
  
  arr = da.random.random(size=size, chunks=chunks)
  y = arr.mean()
  return y.compute()
import time

def dask_arr_chk():
  sizes = []
  times = []
  size = 10
  for i in range(5):
    dim1 = size ** (i+1)
    for j in range(4):
      dim2 = size ** (j+1)
      if dim1*dim2 in sizes: continue
      st = time.time()
      dask_mean(size=(dim1, dim2))
      en = time.time()
      sizes.append(dim1*dim2)
      times.append(en-st)
  return sizes, times

def numpy_arr_chk():
  sizes = []
  times = []
  size = 10
  for i in range(4):
    dim1 = size ** (i+1)
    for j in range(4):
      dim2 = size ** (j+1)
      if dim1*dim2 in sizes: continue
      st = time.time()
      numpy_mean(size=(dim1, dim2))
      en = time.time()
      sizes.append(dim1*dim2)
      times.append(en-st)
  return sizes, times
%%time
x1, y1 = numpy_arr_chk()
x2, y2 = dask_arr_chk()
fig, axs = plt.subplots(1, 3, figsize=(23, 5))
axs[0].plot(x1[:-1], y1[:-1], "o-", label="Numpy")
axs[0].plot(x2[:-2], y2[:-2], "o-", label="Dask")
axs[0].set_xlabel("Array elements:")
axs[0].set_ylabel("Time Taken (sec):")
axs[0].legend()

axs[1].plot(x1, y1, "o-", label="Numpy")
axs[1].plot(x2[:-1], y2[:-1], "o-", label="Dask")
axs[1].set_xlabel("Array elements:")
axs[1].set_ylabel("Time Taken (sec):")
axs[1].legend()

axs[2].plot(x1, y1, "o-", label="Numpy")
axs[2].plot(x2, y2, "o-", label="Dask")
axs[2].set_xlabel("Array elements:")
axs[2].set_ylabel("Time Taken (sec):")
axs[2].legend()
import dask.dataframe as dd
import numpy as np
import gc
gc.enable()
arr = np.random.normal(0.0, 1.0, size=(1000000, 10))
df = dd.from_array(arr, chunksize=50000, columns=[f"col-{i+1}" for i in range(10)])
del arr
gc.collect()
df
df.visualize(size="14,16!")
df.head() # Not lazy beacuse it doesn't take much computation
df.tail()
df["col-1"] = (df["col-1"]*10).astype(int)
agg = df.groupby(by=["col-1"]).aggregate(["sum", "std", "max", "min", "mean"])
agg.head(2)
columns = []
for col in agg.columns.levels[0]:
  for a in agg.columns.levels[1]:
    columns.append(f"{col}.{a}")

agg.columns = columns
agg.head(2)
df_new = df.merge(agg.reset_index(), how="left", on="col-1")
df_new
df_new.visualize(rankdir="LR", size="20, 15!")
df_new.compute().head()
df_new.shape[0].compute(), df_new.shape[1]
import dask.bag as db

lst = []
for i in range(5):
  lst.append({f"Name.{name}": value for name, value in np.random.randint(1, 10, (5, 2))})
  lst.append(np.random.randint(2, 5, (2, 4)))
  lst.append(np.random.randint(1, 1000, (1,)))
  lst.append([i for i in range(100, 200, 10)])
  
b = db.from_sequence(lst)
b.take(1)
def fil(el):
  if type(el)!=dict and type(el)!=list: return True
  else: return False

filmap = b.filter(fil).map(lambda x: x**2)
filmap.visualize(size="15,10!")
filmap.compute()
comp = filmap.flatten().mean()
comp.visualize(size="15, 15!")
comp.compute()
import dask.delayed as delay

@delay
def sq(x):
  return x**2

@delay
def add(x, y):
  return x+y

@delay
def sum(arr):
  sum = 0
  for i in range(len(arr)): sum+=arr[i]
  return sum
# Adding tasks here is like adding nodes to graphs.
# You can add new taks based on results of prevoius tasks.
# Dask won't compute them right away. It will make a graph as
# you call them. And then COmpute the whole graph parallely.
lst = list(np.arange(1, 11))

for i in range(3):
  temp = []
  if i == 0:
    for j in range(0, len(lst)):
      temp.append(sq(lst[j]))
  elif i == 1:
    for j in range(0, len(lst)-1, 2):
      temp.append(add(lst[j], lst[j+1]))
  else:
    temp.append(sum(lst))
  lst = temp # New functions will be computed on last results
  
lst
lst[0].visualize(size="7,10!")
lst[0].compute()
from dask.distributed import Client, LocalCluster # Look into parameters of LocalCluster for arguments used
client = Client(processes=False, threads_per_worker=4, n_workers=4, memory_limit='8GB')
client
def sq(x):
  return x**2

inputs = np.arange(0, 10000000)
sent = client.submit(sq, 1000000)
sent # Pending: Not Complete
sent # Finished (after a few sec): Complete 
result = sent.result()
result
sent = client.submit(sq, inputs,)
sent
sent
sent.result()
from dask_ml.datasets import make_regression
import dask.dataframe as dd

X, y = make_regression(n_samples=1e6, chunks=50000)
df = dd.from_dask_array(X)
df.head()
from dask_ml.model_selection import train_test_split, GridSearchCV

xtr, ytr, xval, yval = train_test_split(X, y)
from sklearn.linear_model import ElasticNet

search_params = {
    "alpha": [.01, .005],
    "l1_ratio": [0.6, 0.8],
    "normalize": [True, False],
}
gsearch = GridSearchCV(ElasticNet(), search_params, cv=10)
#gsearch.fit(X, y)
#gsearch.best_params_
#gsearch.best_score_
from dask_ml.datasets import make_classification
import dask.dataframe as dd

X, y = make_classification(n_samples=1e6, chunks=50000) # number of classes here are 2
from dask_ml.model_selection import train_test_split, GridSearchCV

xtr, xval, ytr, yval = train_test_split(X, y)
from sklearn.linear_model import LogisticRegression

search_params = {
    "C": [.05, .005],
    #"penalty": ["l2", "l1"],
    "class_weight": [None, "balanced"],
    "solver": ["lbfgs"]
}
gsearch = GridSearchCV(LogisticRegression(), search_params, cv=10)
#gsearch.fit(X, y)
#gsearch.best_params_
#gsearch.best_score_
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.externals import joblib

#client = Client() # Not able to make client here
LR = LogisticRegression(C=0.01, class_weight="balanced", penalty="l2", solver="lbfgs")
#with joblib.parallel_backend('dask'):
#    LR.fit(xtr, ytr)
#    preds = LR.predict(xval)
#    
#preds[0:5], yval[0:5]
#preds[0:5], yval.compute()[0:5]
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
xtr.shape, ytr.shape, xval.shape, yval.shape
from dask_ml.cluster import KMeans

KM = KMeans(n_clusters=2)

KM.fit(xtr, ytr)
preds = KM.predict(xval)

preds[0:5], yval[0:5]
preds.compute()[0:5], yval.compute()[0:5]