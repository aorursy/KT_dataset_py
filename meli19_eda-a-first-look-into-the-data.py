# import packages



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# from treeinterpreter import treeinterpreter as ti

pd.set_option('display.float_format', lambda x: '%.5f' % x)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# define series' types



types_dict_train = {'train_id': 'int64',

             'item_condition_id': 'int8',

             'price': 'float64',

             'shipping': 'int8'}



types_dict_test = {'test_id': 'int64',

             'item_condition_id': 'int8',

             'shipping': 'int8'}
train = pd.read_csv('../input/train.tsv',delimiter='\t',low_memory=True,dtype=types_dict_train)

test = pd.read_csv('../input/test.tsv',delimiter='\t',low_memory= True,dtype=types_dict_test)
train.head(3)
test.head(3)
# How many brands ?

print(len(train.brand_name.unique()),"brands")
# How many categories ?

print(len(train.category_name.unique()),"categories")
# How many names of  commodity ?

print(len(train.name.unique()),"commodity names")
# The number of train ID is?

print(len(train.train_id.unique()),"train_ids")
# How many different item descriptions ?

print(len(train.item_description.unique()),"commodity names")
print('the shape of the training data is:')

train.shape
print('the shape of the testing data is:')

train.shape
# train - info

train.info()
# test - info

test.info()
# train - describe

train.describe()
# test - describe

train.describe()
# train - index

train.index
# train - index

test.index
# train - columns

train.columns
# test - columns

test.columns
# set figure size

plt.rcParams["figure.figsize"] = (12,8)

plt.rc('xtick', labelsize=20) 

plt.rc('ytick', labelsize=20)
# Price distribution

_ = sns.distplot(train.price, hist=True, kde=True, bins=100)

plt.xlabel('Price', fontsize=30)

plt.ylabel('Frequency', fontsize=30)

plt.show()
# Price Logarithmic distribution

_ = sns.distplot(np.log1p(train.price), hist=True, kde=True, bins=100)

plt.xlabel('Logarithmic price', fontsize=30)

plt.ylabel('Frequency', fontsize=30)

plt.show()
# Price Logarithmic distribution

_ = sns.distplot(train.item_condition_id, hist=True, kde=True, bins=100)

plt.xlabel('item_condition_id in the training data', fontsize=30)

plt.ylabel('Frequency', fontsize=30)

plt.show()
# Price Logarithmic distribution

_ = sns.countplot(train.shipping)

plt.xlabel('Shipping', fontsize=30)

plt.ylabel('Frequency', fontsize=30)

plt.show()
# item_condition_id vs price

_ = sns.violinplot(x='item_condition_id', y='price', data=train)

plt.xlabel('item_condition_id', fontsize=30)

plt.ylabel('price', fontsize=30)

plt.show()
# item_condition_id vs logarithmic price

_ = sns.violinplot(x='item_condition_id',y=np.log1p(train.price), data=train)

plt.xlabel('item_condition_id', fontsize=30)

plt.ylabel('logarithmic price', fontsize=30)

plt.show()
# shipping vs price

_ = sns.violinplot(x='shipping', y='price', data=train)

plt.xlabel('shipping', fontsize=30)

plt.ylabel('price', fontsize=30)

plt.show()
# shipping vs logarithmic price

_ = sns.violinplot(x='shipping',y=np.log1p(train.price), data=train)

plt.xlabel('shipping', fontsize=30)

plt.ylabel('logarithmic price', fontsize=30)

plt.show()
# any NAs?

train.isnull().sum()
# visualize the missing values

missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()