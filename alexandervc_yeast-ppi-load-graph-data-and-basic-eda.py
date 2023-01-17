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
#from __future__ import division

#from __future__ import print_function



import time





import numpy as np

import scipy.sparse as sp



import networkx as nx

#import tensorflow as tf

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score
def load_data():

    #g = nx.read_edgelist('yeast.edgelist')

    g = nx.read_edgelist('/kaggle/input/yeast-proteinprotein-interaction-network/yeast.edgelist')

    adj = nx.adjacency_matrix(g)

    return adj,g



adj,g = load_data()

num_nodes = adj.shape[0]

num_edges = adj.sum()



print('num_nodes',num_nodes,'num_edges',num_edges)
t0 = time.time()

nx.draw(g)

print(time.time()-t0,'seconds passed')
