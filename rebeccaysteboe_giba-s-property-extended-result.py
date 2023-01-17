%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
DATA_DIR = '../input/'

hex2dec = lambda x: int(x, 16)
train = pd.read_csv(DATA_DIR+'train.csv')
cols = [
    "f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f",
    "fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916",
    "b43a7cfd5","58232a6fb"
]
rows = np.array([2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444])-1
tmp = train.loc[rows, ["ID","target"]+cols]
print('original shape', tmp.shape)
tmp
train_sub = train.loc[rows, :]
train_sub = train_sub.iloc[:, 2:]
train_sub
tmp_new = train.loc[rows, cols]
tmp_new
print('Column searching...')

cnt = 0
flag = True
while flag:
    for c in train_sub.columns:
        if c in tmp_new:
            continue
        elif np.all(
            train_sub[c].iloc[1:].values==tmp_new.iloc[:-1, -1].values
        ) and len(train_sub[c].unique())>1:
            tmp_new[c] = train_sub[c].values
            print(c, 'found!', 'new shape', tmp_new.shape)
            cnt += 1
            break
        else:
            continue
    if cnt==0:
        flag = False
    else:
        flag = True
        cnt = 0
tmp_new
tmp_new_trsps = tmp_new.T.copy()
tmp_new_trsps
train_trsps = train[tmp_new.columns].T.copy()
train_trsps.head()
print('Row searching ...')

cnt = 0
flag = True
while flag:
    for c in train_trsps.columns:
        if c in tmp_new_trsps:
            continue
        elif np.all(
            train_trsps[c].iloc[1:].values==tmp_new_trsps.iloc[:-1, -1].values
        ) and len(train_trsps[c].unique())>1:
            tmp_new_trsps[c] = train_trsps[c].values
            print(c, 'found right!', 'new shape (transposed)', tmp_new_trsps.shape)
            cnt += 1
            break
        elif np.all(
            train_trsps[c].iloc[:-1].values==tmp_new_trsps.iloc[1:, 0].values
        ) and len(train_trsps[c].unique())>1:
            tmp_new_trsps.insert(0, c, train_trsps[c].values)
            print(c, 'found left!', 'new shape (transposed)', tmp_new_trsps.shape)
            cnt += 1
            break
        else:
            continue
    if cnt==0:
        flag = False
    else:
        flag = True
        cnt = 0
tmp_new = tmp_new_trsps.T.copy()
print('new shape', tmp_new.shape)
tmp_new
train_sub = train.loc[tmp_new.index, :]
train_sub = train_sub.iloc[:, 2:]
print(train_sub.shape)
train_sub
print('Column searching (second time) ...')

cnt = 0
flag = True
while flag:
    for c in train_sub.columns:
        if c in tmp_new:
            continue
        elif np.all(
            train_sub[c].iloc[1:].values==tmp_new.iloc[:-1, -1].values
        ) and len(train_sub[c].unique())>1:
            tmp_new[c] = train_sub[c].values
            print(c, 'found!', 'new shape', tmp_new.shape)
            cnt += 1
            break
        else:
            continue
    if cnt==0:
        flag = False
    else:
        flag = True
        cnt = 0
print('new shape', tmp_new.shape)
tmp_new
print(f'Row indexes({tmp_new.shape[0]})\n', tmp_new.index.values.tolist())
print(f'Column indexes({tmp_new.shape[1]})\n', tmp_new.columns.values.tolist())
train.loc[tmp_new.index, ["ID","target"]+tmp_new.columns.tolist()]