import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

#Now let us get rid of the outlier to make it easier to see the differences

train = train[train.y!=train.y.max()]
_ = plt.scatter(train[train.X261==1].ID,train[train.X261==1].y)
_ = plt.scatter(train[train.X261==0].ID,train[train.X261==0].y)
_ = plt.hist(train[train.X261==1].y,bins=100,alpha=.5)

_ = plt.hist(train[train.X261==0].y,bins=100,alpha=.5)