# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline
train = pd.read_csv('../input/train_V2.csv')
train.head()

train.tail()

train.describe()

train.describe(include='O')
train.info()
train.columns
f, ax = plt.subplots(figsize = (20, 15))

sns.countplot(x=train['matchType'], data = train, alpha = 0.5)
crosstab = pd.crosstab(index=train.matchType, columns="count")

crosstab
# f, ax = plt.subplots(figsize = (20, 15))

# sns.distplot(train['assists'], kde=True, rug=True)

# plt.title("Assists boxplot")



skip_col = ['matchType', 'Id', 'groupId', 'matchId']

for c in train.columns:

    if c in skip_col:

        continue

    print("-" * 50)

    print("column : ", c)

    

    f, ax = plt.subplots(figsize = (20, 15))

    sns.distplot(train[c], kde = True, rug = True)

    plt.title(c + "distplot")

    plt.show()