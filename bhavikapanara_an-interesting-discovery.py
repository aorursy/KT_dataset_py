import pandas as pd

pd.set_option('display.max_columns', 50) 

import numpy as np

import seaborn as sns
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



train.shape,test.shape
train.head()
train['nom_0'] = train['nom_0'].astype(str)

test['nom_0'] = test['nom_0'].astype(str)



train['nom0__ord1'] = train[['nom_0','ord_1']].apply(''.join, axis=1)

test['nom0__ord1'] = test[['nom_0','ord_1']].apply(''.join, axis=1)
train.head()
one_hot = pd.get_dummies(train['nom0__ord1'])

one_hot.shape
one_hot.head()