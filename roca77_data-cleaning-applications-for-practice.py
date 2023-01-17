# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import nan as NA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df
df.fillna(0)
df.fillna({1: 0.5, 2: 0})
df = pd.DataFrame(np.random.randn(6, 3))
df.iloc[2:, 1] = NA
df.iloc[4:,2] = NA
df
df.iloc[:, 1]
df.iloc[[2, 3, 4], [1, 2]]
df = df.fillna(method='ffill')
df
df.fillna(method='ffill', limit=2)
data = pd.Series([1., NA, 3.5, NA, 7])
data
data.fillna(data.mean())
data = pd.DataFrame(np.arange(12).reshape((3, 4)), 
                   index=['Ohio', 'Colorado', 'New York'],
                   columns=['one', 'two', 'three', 'four'])
data
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                    'foo', 'foo', 'qux', 'qux'],
                   ['one', 'two', 'one', 'two',
                   'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
tuples
index
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df
df2 = df[:4]
df2
stacked = df2.stack()
stacked
stacked.unstack()
stacked.unstack(1)
stacked.unstack(0)
columns = pd.MultiIndex.from_tuples([
    ('A', 'cat', 'long'), ('B', 'cat', 'long'),
    ('A', 'dog', 'short'), ('B', 'dog', 'short')], 
names=['exp', 'animal', 'hair_length'])
columns
df = pd.DataFrame(np.random.randn(4, 4), columns=columns)
df
df.stack(level=['animal', 'hair_length'])
df.stack(level=[1, 2])
columns = pd.MultiIndex.from_tuples([('A', 'cat'), ('B', 'dog'),
                                    ('B', 'car'), ('A', 'dog')],
                                   names=['exp', 'animal'])
index = pd.MultiIndex.from_product([('bar', 'baz', 'foo', 'qux'), 
                                   ('one', 'two')],
                                  names=['first', 'second'])
columns
index
