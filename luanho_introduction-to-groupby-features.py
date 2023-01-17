import numpy as np
import pandas as pd 
import os

df = pd.DataFrame({'key1' : ['a','a','b','b','a'],
                    'key2' :['one','two','one','two','one'] ,
                    'data1': np.random.rand(5),
                    'data2':np.random.rand(5)
                  })
df
grouped_by_key1 = df['data1'].groupby(df['key1'])

print(grouped_by_key1.mean())
print(grouped_by_key1.max())
print(grouped_by_key1.median())
print(grouped_by_key1.size())
print(grouped_by_key1.min())
grouped = df.groupby(['key1','key2'])
print(grouped.mean())
grouped.mean().unstack()
k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means
pd.merge(df , k1_means, left_on='key1', right_index = True)
df.groupby('key2').transform(np.mean)
def demean(arr):
    return arr - arr.mean() 
df.groupby('key2').transform(demean)
train = pd.read_csv('../input/train.csv')
train.head()
def top(data, n, column):
    return data.sort_index(by=column)[:n]

train.groupby('Sex').apply(top, n=5, column='Fare')
train.groupby('Embarked').apply(top, n=5, column='Fare')