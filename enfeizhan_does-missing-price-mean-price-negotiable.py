# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.shape
train.isnull().sum()
train.isnull().sum() / train.shape[0] * 100
train.describe()
fig, ax = plt.subplots(figsize=(15, 8))
train.loc[train.price < 7e3, ['price', 'deal_probability']].plot(kind='scatter', x='price', y='deal_probability', ax=ax, alpha=0.1, color='r')
ax.grid()
plt.show()
fig, ax = plt.subplots(figsize=(15, 8), nrows=2)
train.loc[train.price.isnull(), 'deal_probability'].plot(kind='hist', bins=20, ax=ax[0], color='r', grid=True)
ax[0].set_title('No price')
train.loc[train.price < 7e3, 'deal_probability'].plot(kind='hist', bins=20, ax=ax[1], color='b', grid=True)
ax[1].set_title('Price < 7,000')
ax[1].set_xlabel('Deal probability')
plt.show()
test = pd.read_csv('../input/test.csv')
test.isnull().sum() / test.shape[0] * 100
