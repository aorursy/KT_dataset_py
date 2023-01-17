# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
print(train.shape)

print(test.shape)
train['comment_text'].value_counts().head()
train[train['target'] >= 0.5]['comment_text'].value_counts().head()
train_size = train.shape[0]

print(f'Above zero rate: {(train["target"] > 0).sum() / train_size:.2%}')

print(f'Toxic rate: {(train["target"] >= 0.5).sum() / train_size:.2%}')

print(f'100% Toxic rate: {(train["target"] == 1.0).sum() / train_size:.2%}')
num_bins = 20

a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

ax.set_xticks(np.linspace(0,1,num_bins+1))



sns.distplot(train['target'], kde=False, ax=ax, bins=num_bins)
train['toxic'] = train['target'] >= 0.5
train['date'] = pd.to_datetime(train['created_date']).dt.date
plot_data = train.groupby('date')['toxic'].agg(['mean', 'count'])

plot_data = plot_data / plot_data.sum()
plot_data.columns
a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

sns.lineplot(data=plot_data, ax=ax)