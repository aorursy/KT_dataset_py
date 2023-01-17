!pip install deepwalk

!pip install pyclustertend
## libraries 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import subprocess

import shlex

import deepwalk

import networkx as nx

import matplotlib.pyplot as plt

from networkx.readwrite.edgelist import read_edgelist

from networkx.readwrite.edgelist import write_edgelist



from sklearn.preprocessing import normalize

from sklearn.semi_supervised import LabelPropagation, LabelSpreading
!deepwalk --format edgelist --input /kaggle/input/bitcoin-to-edge-list/bitcoin_edge.edgelist --workers 10 --number-walks 20 --representation-size 128 --walk-length 30 --window-size 5 --output /kaggle/working/graph_bitcoin.embeddings
_embeddings = np.loadtxt('/kaggle/input/bitcoin-graph-embedding/graph_bitcoin.embeddings', skiprows = 1)
df_classes = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
df_classes['class'].value_counts()
_embeddings
_embeddings.shape
df_classes['class'][df_classes['class'] == 'unknown'] = -1
y = df_classes['class'].to_numpy()

y = y.astype(int)
model = LabelSpreading(kernel = 'knn', max_iter = 10000, tol = 0.3, n_jobs = -1)
normalized_X = normalize(_embeddings)
model.fit(normalized_X,y)
result_label_propagation = model.predict_proba(normalized_X)
plt.hist(result_label_propagation[:,1], color = 'blue', edgecolor = 'black')