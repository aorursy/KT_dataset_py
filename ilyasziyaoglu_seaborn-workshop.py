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
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/kc_house_data.csv')

df.head()
len(df) - df.count()
sns.barplot(x="bedrooms", y="price", data=df, estimator=np.median, capsize=0.05, ci=100, hue='waterfront', palette='winter')
sns.barplot(x="bedrooms", y="price", data=df, estimator=np.median, capsize=0.05, ci=100, hue='waterfront', palette='winter', order=[3,2,1,0])
sns.distplot(df.price)
sns.boxenplot(df.price)
sns.boxenplot(x = 'bedrooms', y = 'price', data=df, order=[0,1,2,3])
sns.boxenplot(x = 'price', y = 'bedrooms', data=df, hue='waterfront', order=[0,1,2,3], orient='h', palette='husl')
sns.swarmplot(x='bedrooms', y='price', data=df[:200])
sns.stripplot(df.price, orient='v', jitter=0.01, marker='D')
sns.stripplot(x='bedrooms', y='price', data=df, order=[0,1,2,3], jitter=True, size=5, hue='waterfront', split=True, edgecolor='gray', alpha=0.2)
sns.stripplot(x='bedrooms', y='price', data=df, order=[0,1,2,3], size=2, jitter=0.02)

sns.boxenplot(x='bedrooms', y='price', data=df, order=[0,1,2,3])
sns.stripplot(x='price', y='bedrooms', data=df, order=[0,1,2,3], size=2, jitter=0.02, orient='h')

sns.violinplot(x='price', y='bedrooms', data=df, order=[0,1,2,3], orient='h', color='0.9')
grid = sns.PairGrid(df[:1000])

grid = grid.map(plt.scatter)
grid = sns.PairGrid(df[:1000])

grid = grid.map_diag(plt.hist)

grid = grid.map_offdiag(plt.scatter)
grid = sns.PairGrid(df[['price', 'bedrooms', 'grade', 'yr_built']].loc[:1000], hue='grade')

grid = grid.map_diag(plt.hist)

grid = grid.map_offdiag(plt.scatter)
grid = sns.PairGrid(df[['price', 'bedrooms', 'grade', 'yr_built']].loc[:1000], hue='grade')

grid = grid.map_diag(plt.hist)

grid = grid.map_offdiag(plt.scatter)

grid = grid.add_legend()
grid = sns.PairGrid(df[:1000], hue='grade', palette='RdBu', vars=['price', 'bedrooms', 'grade', 'yr_built'])

grid = grid.map_diag(plt.hist, histtype='step', linewidth=2.5)

grid = grid.map_offdiag(plt.scatter)

grid = grid.add_legend()
grid = sns.PairGrid(df[['price', 'bedrooms', 'grade', 'yr_built']].loc[:1000], hue='grade', palette='RdBu')

grid = grid.map_diag(plt.hist, histtype='step', linewidth=2.5)

grid = grid.map_offdiag(plt.scatter)

grid = grid.add_legend()
df.head()
df.info()