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
DATASET_ORIGINAL = "/kaggle/input/stanford/web-Stanford.txt" #Uncompress the .gz file! 
NODES = 281903 
EDGES = 2312497

import numpy as np

from  scipy import sparse
def dataset2dok():
    with open(DATASET_ORIGINAL,'r') as f:
        dokm = sparse.dok_matrix((NODES,NODES),dtype=np.bool)
        for line in f.readlines()[4:]:
            origin, destiny = (int(x)-1 for x in line.split())
            dokm[destiny,origin]=True
    return(dokm.tocsr())

%time dok_m = dataset2dok()
def dataset2csr():
    row = []
    col = []    
    with open(DATASET_ORIGINAL,'r') as f:
        for line in f.readlines()[4:]:
            origin, destiny = (int(x)-1 for x in line.split())
            row.append(destiny)
            col.append(origin)
    return(sparse.csr_matrix(([True]*EDGES,(row,col)),shape=(NODES,NODES)))

%time csr_m = dataset2csr()
import sys

print ("The size in memory of the adjacency matrix is {0} MB");
format(
    (sys.getsizeof(csr_m.shape)+
    csr_m.data.nbytes+
    csr_m.indices.nbytes+
    csr_m.indptr.nbytes)/(1024.0**2)
)

def csr_save(filename,csr):
    np.savez(filename,
        nodes=csr.shape[0],
        edges=csr.data.size,
        indices=csr.indices,
        indptr =csr.indptr
    )

def csr_load(filename):
    loader = np.load(filename)
    edges = int(loader['edges'])
    nodes = int(loader['nodes'])
    return sparse.csr_matrix(
        (np.bool_(np.ones(edges)), loader['indices'], loader['indptr']),
        shape = (nodes,nodes)
    )
DATASET_NATIVE = 'dataset-native.npz'
csr_save(DATASET_NATIVE,csr_m)
%time csr = csr_load(DATASET_NATIVE)
def compute_PageRank(G, beta=0.85, epsilon=10**-4):
     
# Матрица смежности тестов в порядке
    n,_ = G.shape
    assert(G.shape==(n,n))
    #Константы  Speed-UP
    deg_out_beta = G.sum(axis=0).T/beta #вектор
    #инициализация
    ranks = np.ones((n,1))/n #вектор
    time = 0
    flag = True
    while flag:        
        time +=1
        with np.errstate(divide='ignore'): # Игнорировать деление на 0 в рангах / deg_out_beta
            new_ranks = G.dot((ranks/deg_out_beta)) #вектор
        #пропущенные PageRank
        new_ranks += (1-new_ranks.sum())/n
        #Условие остановки
        if np.linalg.norm(ranks-new_ranks,ord=1)<=epsilon:
            flag = False        
        ranks = new_ranks
    return(ranks, time)

print ('==> Computing PageRank');
%time pr,iters = compute_PageRank(csr)
print ('\nIterations: {0}'.format(iters))
print ('Element with the highest PageRank: {0}'.format(np.argmax(pr)+1))
import networkx as nx

print ('==> Loading data.')
with open(DATASET_ORIGINAL,'r') as f:
    edgelist = [
        tuple(int(x)-1 for x in line.split())
        for line in f.readlines()[4:]
    ] 
    
print ('\n==> Building graph.')
%time g = nx.from_edgelist(edgelist, create_using=nx.DiGraph())

print ('\n==> Computing PageRank')
%time pr = nx.pagerank(g)

pr_max = max(pr.items(), key= lambda x: x[1])
print ('\nElement with the highest PageRank: {0}'.format(pr_max[0]+1))
rr = max(nx.pagerank(g))
print (' the highest PageRank is', rr)