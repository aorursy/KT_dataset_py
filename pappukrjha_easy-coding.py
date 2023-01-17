# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os, sys



pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/train_2016.csv')
data = train

print(data.head())

print(data.shape)

tmp = pd.DataFrame(data.dtypes).reset_index()

tmp.columns = ['columns', 'dtypes']

print(tmp.groupby('dtypes')['columns'].size())

print(data.isnull().sum())

print(data['parcelid'].nunique())

tmp = data.groupby('parcelid')['transactiondate'].size().reset_index()

tmp.columns = ['parcelid','freq']

print(tmp.query('freq>1').shape)#properties sold more than once

tmp.sort('freq', ascending = False, inplace = True)

print(tmp.head())
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
plt.scatter(train['logerror'].sort_values(), train.index.values)

plt.show()
plt.hist(train['logerror'])

plt.show()
tmp = train.groupby('transactiondate')['logerror'].sum().reset_index()

tmp.columns = ['transactiondate', 'logerror']

print(tmp.head())
plt.scatter(tmp['logerror'].sort_values(), tmp.index.values)

plt.show()
prop2016 = pd.read_csv('../input/properties_2016.csv')
data = prop2016

print(data.head())

print(data.shape)

tmp = pd.DataFrame(data.dtypes).reset_index()

tmp.columns = ['columns', 'dtypes']

print(tmp.groupby('dtypes')['columns'].size())

print(data.isnull().sum())

print(data['parcelid'].nunique())
df_train = pd.merge(train, prop2016, on = 'parcelid', how = 'left')
df_train['logerror'] = df_train['logerror'].map(float)

def pruneOutlier(x):

    if x < -2 :

        return -2

    elif x > 2 :

        return 2

    else :

        return x

df_train['logerror'] = df_train['logerror'].map(lambda x : pruneOutlier(x))
plt.scatter(df_train['logerror'].sort_values(), train.index.values)

plt.show()