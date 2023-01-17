import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('../input/train.csv')

train.drop('ID',inplace=True,axis=1)

train.drop('y',inplace=True,axis=1)

train.drop('X5',inplace=True,axis=1)

train = train.T.drop_duplicates().T

print(train.shape)

train.drop_duplicates(inplace=True)

print(train.shape)
test = pd.read_csv('../input/test.csv')

test.drop('ID',inplace=True,axis=1)

test.drop('X5',inplace=True,axis=1)

test = test[train.columns[2:]]

test = test.T.drop_duplicates().T

print(test.shape)

test.drop_duplicates(test.columns[1:],inplace=True)

print(test.shape)
train = train[test.columns]
alldata = pd.concat([train,test])

print(alldata.shape)

alldata.drop_duplicates(inplace=True)

print(alldata.shape)