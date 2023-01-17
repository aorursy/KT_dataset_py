# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test_df = pd.read_csv('../input/gene-expression/data_set_ALL_AML_independent.csv')
test_df.head()
test_df = test_df.drop(['Gene Description','Gene Accession Number'],axis=1)
call_cols = ['call']+['call.'+str(i) for i in range(1,34)]
test_df = test_df.drop(call_cols,axis=1)
test_df.columns
test_x = test_df.as_matrix().T
actual_df = pd.read_csv('../input/gene-expression/actual.csv')
test_y = actual_df['cancer'][38:]
test_y.shape
from sklearn.preprocessing import MinMaxScaler as scaler
min_scaler = scaler()
test_x_norm = min_scaler.fit_transform(test_x)
def eu_dist(g1,g2):
    return 1 - (g1-g2)*(g1-g2)
test_x1 = np.zeros([test_x.shape[1],test_x.shape[1]])
for i in range(test_x1.shape[0]):
    for j in range(test_x1.shape[0]):
        if i!=j :
            test_x1[i,j] = eu_dist(test_x_norm[0][i],test_x_norm[0][j])
    print(i)
np.mean(test_x1)
np.sum(test_x1>=0.999)/(7219*7219)
test_x1[test_x1<0.999] = 0
G_x1 = nx.convert_matrix.from_numpy_matrix(test_x1)
print(nx.info(G_x1))
def degree_cen(G):
    degs = {}
    for v in G.nodes():
        degs[v] = G.degree(v)/(G.number_of_nodes()-1)
        if int(v)%100==0:
            print(v)
    return degs
def cluster_coef(G):
    clus_coef = {}
    count =  0
    for u in G.nodes():
        clus_coef[u] = nx.clustering(G,u)
        print(count)
        count+=1
    return clus_coef
deg_list_x1 = degree_cen(G_x1)
level_x1_1 = sorted(deg_list_x1, key=deg_list_x1.get, reverse=True)[:1000]
len(level_x1_1)
G_x1_1 = G_x1.subgraph(level_x1_1)
clus_coef_x1 = cluster_coef(G_x1_1)
clus_coef_x1new = sorted(clus_coef_x1, key=clus_coef_x1.get, reverse=True)[:100]
clus_df1 = pd.read_csv('../input/gene-expression-data-analysis-sn-1-1/clus_ALL.csv')
clus_df2 = pd.read_csv('../input/gene-expression-data-analysis-sn-1-1/clus_AML.csv')
level_1 = list(clus_df1['0'].values)
len(set(level_x1_1).intersection(level_1))
level_12 = list(clus_df2['0'].values)
len(set(level_x1_1).intersection(level_12))
test_x11 = np.zeros([test_x.shape[1],test_x.shape[1]])
for i in range(test_x11.shape[0]):
    for j in range(test_x11.shape[0]):
        if i!=j :
            test_x11[i,j] = eu_dist(test_x_norm[11][i],test_x_norm[11][j])
    print(i)
np.mean(test_x11)
np.sum(test_x11>=0.999)/(7219*7219)
test_x11[test_x11<0.999] = 0
G_x11 = nx.convert_matrix.from_numpy_matrix(test_x11)
print(nx.info(G_x11))
deg_list_x11 = degree_cen(G_x11)
level_x11_1 = sorted(deg_list_x11, key=deg_list_x11.get, reverse=True)[:1000]
len(level_x1_1)
level_1 = list(clus_df1['0'].values)
len(set(level_x11_1).intersection(level_1))
len(set(level_x11_1).intersection(level_12))
len(set(level_1).intersection(level_12))
