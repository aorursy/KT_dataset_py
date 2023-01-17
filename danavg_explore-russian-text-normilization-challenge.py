import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns
train = pd.read_csv('../input/ru_train.csv')

test = pd.read_csv('../input/ru_test_2.csv')
train.head()
test.head()
train.shape
test.shape
train['class'].unique()
train[train['class'] == 'PLAIN'].head(3)
train[train['class'] == 'DATE'].head(3)
train[train['class'] == 'PUNCT'].head(3)
train[train['class'] == 'ORDINAL'].head(3)
train[train['class'] == 'VERBATIM'].head(3)
train[train['class'] == 'LETTERS'].head(3)
train[train['class'] == 'CARDINAL'].head(3)
train[train['class'] == 'MEASURE'].head(3)
train[train['class'] == 'TELEPHONE'].head(3)
train[train['class'] == 'ELECTRONIC'].head(3)
train[train['class'] == 'DECIMAL'].head(3)
train[train['class'] == 'DIGIT'].head(3)
train[train['class'] == 'FRACTION'].head(3)
train[train['class'] == 'MONEY'].head(3)
train[train['class'] == 'TIME'].head(3)
dict1 = dict(Counter(train['class']))

dict1
class1 = list(train['class'].unique())
names = list(dict1.keys())

values = list(dict1.values())
fig, ax = plt.subplots(1,1, figsize=(10,12))

count_classes_fig = sns.countplot(y="class", data=train, ax=ax)

for item in count_classes_fig.get_xticklabels():

    item.set_rotation(45)