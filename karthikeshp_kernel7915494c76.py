# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
canes = pd.read_csv('/kaggle/input/sugarcane-disease-data/cane.csv')

canes.head()
canes.columns = ['SNo', 'n', 'r', 'x', 'var', 'block']
canes.describe(include='all').T
canes.SNo.plot()
canes.set_index('SNo', inplace=True)
canes.block.value_counts()
sns.pairplot(canes)
canes.pivot_table(index='block', aggfunc=['min', 'max', 'mean'])
sns.violinplot(data=canes, x='block', y='n')
sns.violinplot(data=canes, x='block', y='var')
sns.violinplot(data=canes, x='block', y='r')
sns.violinplot(data=canes, x='block', y='x')
fig, ax = plt.subplots(figsize=(15, 10))

sns.violinplot(data=canes)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

sns.violinplot(data=canes, x='var', y='n', ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

sns.violinplot(data=canes, x='var', y='r', ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

sns.violinplot(data=canes, x='var', y='x', ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

sns.violinplot(data=canes, x='var', y='n', ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(canes.corr(), ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(20, 10))

sns.lineplot(data=canes, x='n', y='r')

plt.show()