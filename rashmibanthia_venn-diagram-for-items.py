# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib_venn import venn2

test = pd.read_csv('../input/test.csv')

items = pd.read_csv('../input/items.csv')



types = {'id': 'int64',

                'item_nbr': 'int32',

                'store_nbr': 'int8',

                'unit_sales': 'float32',

                'onpromotion': 'object',

            }

train = pd.read_csv('../input/train.csv', parse_dates = ['date'], dtype = types, 

                       infer_datetime_format = True)
train.head(n=2)
test.head(n=2)
test_items = pd.unique(test.item_nbr)

train_items = pd.unique(train.item_nbr)

items_in_train_and_test  = set(test_items).intersection(set(train_items))

items_only_in_test = set(test_items).difference(set(train_items))

items_only_in_train = set(train_items).difference(set(test_items)) #not in test
#Some adjustment done for better visualization, else we would get overlapping circles.

v = venn2(subsets = (len(train_items), len(items_in_train_and_test), len(test_items)), 

      set_labels=('Train', 'Test'));

# Subset labels

v.get_label_by_id('10').set_text('195')

v.get_label_by_id('01').set_text('60')

v.get_label_by_id('11').set_text('3841')

print(f'There are {len(test_items)} distinct items in test.')

print(f'There are {len(train_items)} distinct items in train.')

print(f'There are {len(items_in_train_and_test)} items common in train and test.')

tmp = np.round((100*test[test.item_nbr.isin(items_only_in_test)].shape[0]/test.shape[0]),2)

print(f'There are {len(items_only_in_test)} items only in test data (not in train.)', 

      f'These items represent about {tmp}% of test data.')

tmp = np.round((100*train[train.item_nbr.isin(items_only_in_train)].shape[0]/train.shape[0]),2)

print(f'There are {len(items_only_in_train)} items only in train data (not in test.)',

      f'These items represent about {tmp}% of train data.')

items_not_anywhere = set(items.item_nbr).difference(items_only_in_test.union(items_only_in_train).union(items_in_train_and_test))

print('Following items not in train or test:')

items[items.item_nbr.isin(items_not_anywhere)]