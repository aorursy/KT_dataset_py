import os
import sys
import pandas as pd
import pickle as pkl
import numpy as np
import gc
from pandas.core.common import array_equivalent
from sklearn.cluster import k_means
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

print('Modules loaded')
#--read data
print('Reading data')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print('Original shapes:', train.shape,test.shape)
#--
#--drop ID and TARGET
train.drop('ID',axis=1,inplace=True)
testIds = test.ID
test.drop('ID',axis=1,inplace=True)
trainY = train.TARGET
train.drop('TARGET',axis=1,inplace=True)
print('Shapes after dropping ID and TARGET:', train.shape,trainY.shape,test.shape)
#pkl.dump(testIds,open('testIds.pkl','wb'))
#--
#--add zero count per row--
def zero_count(x):
    return sum(x==0)
train['zero_count'] = train.apply(zero_count,axis=1)
test['zero_count'] = test.apply(zero_count,axis=1)
print('Shapes after adding zero_count:', train.shape,trainY.shape,test.shape)
#---
#--remove constant vars--
constant_vars = list(train.columns[train.apply(pd.Series.nunique) == 1])
print('No. of constant vars in train:', len(constant_vars))
train.drop(constant_vars,axis=1,inplace=True)
test.drop(constant_vars,axis=1,inplace=True)
print('Shapes after dropping constant vars in train:', train.shape,trainY.shape,test.shape)
#--
#--drop duplicate columns--
def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups
dup_cols = duplicate_columns(train)
print('No. of duplicate cols:', len(dup_cols))
train = train.drop(dup_cols, axis=1)
test = test.drop(dup_cols, axis=1)
print('Shapes after dropping duplicate columns:', train.shape,trainY.shape,test.shape)
#--
corr = train.corr()
print('Correlation mat is calculated')
n_clusters = 140
w_ss_arr = []
km_arr = []
def get_w_ss(centroids,idx):
    w_ss = 0
    for uidx in np.unique(idx):
        d = euclidean_distances(corr[idx==uidx],[list(centroids[uidx])])
        w_ss = w_ss+np.sum(np.square(np.squeeze(d)))
    return w_ss

for i in range(140,n_clusters+1):
    print('Clustering:', i)
    km = k_means(corr,i)
    w_ss_arr.append(get_w_ss(km[0],km[1]))
    km_arr.append(km)

plt.plot(range(140,n_clusters+1),w_ss_arr)
plt.show()
#---
idxs = km_arr[9][1] #---n_clusters = 139 seems to be the point of saturation on the scree plot

col_idx = np.arange(train.shape[1])
rearranged_col_idx = []

for unique_idx in np.unique(idxs):
    rearranged_col_idx = rearranged_col_idx + list(col_idx[idxs==unique_idx])

#train = train.ix[:,rearranged_col_idx]
train = train.ix[:,np.array(train.columns)[rearranged_col_idx]]
corr = train.corr()

print('Re-arranged correlation matrix calculated')

#plot heatmap
sns.set(context="paper", font="monospace")
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True)
f.tight_layout()