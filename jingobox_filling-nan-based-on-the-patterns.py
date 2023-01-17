import numpy as np

import pandas as pd

from itertools import permutations

import time

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
%%time

train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
#train

tr1 = train[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/train.shape[0]

tr1.plot(kind='barh')
#test

te1 = test[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/test.shape[0]

te1.plot(kind='barh')
%%time

cols = ['D{}'.format(j) for j in range(1,16) if j not in [8,9]]



r = list(range(0,len(cols)))

l = list(permutations(r,2))

data = pd.concat([train[['C1','C2','C13','C14'] + cols],test[['C1','C2','C13','C14'] + cols]] )



print(data[data==0].count().sum())



for q in l:



    data['count'] = 0

    x = data.groupby(['C1','C2','C13','C14',cols[q[0]]])['count'].count().reset_index()

    x.sort_values(['count','C1','C2','C13','C14',cols[q[0]]], ascending=False,inplace=True)



    for i in range(0,500):

        

           if x.iloc[i,4] == 0:



                if data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]].value_counts(dropna=False).shape[0] == 2:



                    if data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]].value_counts().shape[0] == 1:

                        

                        val = data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]].max()

                   

                        data.loc[(data['C1']==x.iloc[i,0]) & (data['C2']==x.iloc[i,1]) & (data['C13']==x.iloc[i,2]) & (data['C14']==x.iloc[i,3]) & (data[cols[q[0]]]==x.iloc[i,4]),cols[q[1]]] = val

                     

                        if x.iloc[i,5]<100: 

                            

                              break



print(data[data==0].count().sum())



train = data[:590540]

test = data[590540:]

del data
#train

tr2 = train[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/train.shape[0]

tr1['after filling NaN'] = tr2

tr1.plot(kind='barh')
#test

te2 = test[['D{}'.format(i) for i in range(1,16) if i not in [8,9]]].notnull().sum().to_frame()/test.shape[0]

te1['after filling NaN'] = te2

te1.plot(kind='barh')