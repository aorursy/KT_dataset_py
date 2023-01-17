import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.info()
test_df.head()
test_df.info()
train_df.columns.groupby(train_df.dtypes)
def count_nulls(df):
    null_counter = df.isnull().sum(axis=0)
    null_counter = null_counter[null_counter > 0]
    null_percent = df.isnull().sum(axis=0) / df.shape[0] * 100
    null_percent = null_percent[null_percent > 0]
    null_df = pd.concat([null_counter,null_percent],axis=1)
    null_df.columns = ['count','percent']
    display(null_df)
count_nulls(train_df)
count_nulls(test_df)
exclude_cols = ['Id','Target','v2a1','v18q1','rez_esc']
np.unique(train_df['Target'])
train_df['Target'].value_counts()
plt.hist(train_df['Target'])
plt.show()
print([x for x in train_df.columns if train_df[x].dtype=='O'])
train_df.idhogar = train_df.idhogar.astype('category')
test_df.idhogar = test_df.idhogar.astype('category')
train_df.dependency.value_counts()
train_df['dependency_calculated'] = (train_df.hogar_nin + train_df.hogar_mayor)/(train_df.hogar_adul - train_df.hogar_mayor)
train_df[['dependency','dependency_calculated']]
train_df.dependency.replace('no','0',inplace=True)
train_df.dependency.replace('yes','1',inplace=True)
train_df.dependency_calculated.replace(float('inf'),8,inplace=True)
all(np.isclose(train_df.dependency.astype('float'), train_df.dependency_calculated))
test_df.dependency.replace('no','0',inplace=True)
test_df.dependency.replace('yes','1',inplace=True)
train_df.dependency = train_df.dependency.astype('float')
test_df.dependency = test_df.dependency.astype('float')
train_df.drop('dependency_calculated', axis=1, inplace=True)
train_df.edjefe.value_counts()
train_df.edjefe.replace('no','0',inplace=True)
train_df.edjefe.replace('yes','1',inplace=True)
train_df.edjefe = train_df.edjefe.astype('float')
test_df.edjefe.replace('no','0',inplace=True)
test_df.edjefe.replace('yes','1',inplace=True)
test_df.edjefe = test_df.edjefe.astype('float')

train_df.edjefa.replace('no','0',inplace=True)
train_df.edjefa.replace('yes','1',inplace=True)
train_df.edjefa = train_df.edjefa.astype('float')
test_df.edjefa.replace('no','0',inplace=True)
test_df.edjefa.replace('yes','1',inplace=True)
test_df.edjefa = test_df.edjefa.astype('float')
exclude_cols
use_cols = train_df.columns.difference(exclude_cols)
print(len(use_cols))
print(use_cols)
for x in use_cols.difference(['idhogar']):
    train_df[x] = train_df[x].astype('float')
    test_df[x] = test_df[x].astype('float')
train_df[use_cols].dtypes.value_counts()
test_df[use_cols].dtypes.value_counts()