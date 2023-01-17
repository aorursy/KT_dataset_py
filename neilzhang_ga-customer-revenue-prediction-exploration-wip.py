# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import ast
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv', dtype=object)
train = pd.read_csv('../input/train.csv', dtype=object)
test.head(1)
ids = test['fullVisitorId'].values
idss = ids.astype('str')
len(ids), type(ids)
uniq_ids = np.unique(idss)
num_ids = len(uniq_ids) #Unique IDs in the test.csv
len(uniq_ids)
#Create a dummy but valid submission
#d = {'fullVisitorId':uniq_ids,'PredictedLogRevenue':np.zeros(num_ids)}
#df = pd.DataFrame(d)
#df.to_csv('./submission_all_zeros.csv', index=False)
#print(os.listdir("."))
idx = 5
s = train.iloc[idx]
print(train.columns)
for c in train:
    print(train[c][:5])
totals = train['device'][1]
totals
#TEST
s = set()
s.update([1,2,2])
print(s, len(s))
print(type(totals))
s = json.loads(totals)
print(type(s), s)
#ast.literal_eval(totals)
#df.apply (dict_to_cols,axis=1)
def expand_tree_column(col_data):
    keys = set()
    ds = []
    for cd in col_data:
        d = json.loads(cd)
        ds.append(d)
        keys.update(list(d.keys()))
    #print('unique keys: %s', keys)
    res = {k:[] for k in keys}
    for d in ds:
        for k in res.keys():
            res[k].append(d.get(k, None))
    return res
def expand_col(df, col_name):
    col_data = np.array(df[col_name])
    res = expand_tree_column(col_data)
    return res
#Expand dictionary-like columns in train.csv
dict_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
traine = train.copy()
for dict_col in dict_cols:
    print('processing column: {}'.format(dict_col))
    res = expand_col(train, dict_col)
    traine = traine.drop(dict_col, axis=1)
    for k, vec in res.items():
        traine[k] = vec
traine.to_csv('./train_expanded.csv', index=False)
#Expand dictionary-like columns in test.csv
teste = test.copy()
for dict_col in dict_cols:
    print('processing column: {}'.format(dict_col))
    res = expand_col(test, dict_col)
    teste = teste.drop(dict_col, axis=1)
    for k, vec in res.items():
        teste[k] = vec
teste.to_csv('./test_expanded.csv', index=False)
#TEST
dd = {'a':1, 'b':2, 'c':3}
for d,v in dd.items():
    print(d,v)

print(len(traine), len(train), len(teste), len(test))
print(os.listdir("./"))
traine.head(5)
teste = pd.read_csv('./test_expanded.csv', dtype=object)
traine = pd.read_csv('./train_expanded.csv', dtype=object)
trs = np.array(traine['transactionRevenue'])
trs = [0 if np.isnan(float(tr)) else float(tr) for tr in trs]
trsn = [np.log10(tr) for tr in trs if tr > 0]
trsnn = [tr for tr in trsn if tr >=7 and tr < 9]
len(trsnn), len(trsn)
len(trsnn)/len(trsn)/2
a,b,c= plt.hist(trsn, density=1, bins=[4,5,6,7,9,10,11])
a, b, c
c[0]
10**8,10**7
