# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# first 3 segments

train = pd.read_csv('../input/train.csv', nrows=450000)

pd.options.display.precision

pd.options.display.precision = 18

train.head(150000) #.info()

import seaborn as sns

sns.set()

# 1st segment

ax = sns.lineplot(x=np.arange(150000), y=train['acoustic_data'][:150000])

# 2nd segment

ax = sns.lineplot(x=np.arange(150000), y=train['acoustic_data'][150000:300000])

# 3rd segment

ax = sns.lineplot(x=np.arange(150000), y=train['acoustic_data'][300000:450000])

train.dtypes #.value_counts()

train.nunique() #.sort_values()

train.acoustic_data.value_counts()

pd.options.display.max_rows

pd.options.display.max_rows = 250

train.acoustic_data.value_counts().sort_index(ascending=False)

ax = sns.lineplot(train.acoustic_data.value_counts().index, train.acoustic_data.value_counts())

ax = sns.distplot(train.acoustic_data, bins=20, kde=False)

train.duplicated().sum()

train.isna().sum() #.sort_values(ascending=False)

train.eq(0).sum() #.sort_values(ascending=False)

train.lt(0).sum() #.sort_values(ascending=False)

train.min() #.sort_values(ascending=False)

train.max() #.sort_values(ascending=False)

del train

# all targets

rows = 150000

target = pd.read_csv('../input/train.csv', usecols=[1]).iloc[rows-1::rows, 0]

ax = sns.lineplot(x=np.arange(target.size), y=target)

target.diff().gt(0).sum()

target.mean()

target.min()

target.max()

ax = sns.distplot(target, bins=20, kde=False)

ax = sns.boxplot(target)
