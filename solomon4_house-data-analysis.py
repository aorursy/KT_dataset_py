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
df = pd.read_csv('/kaggle/input/data.csv')

df.columns
df.describe()
df = df.dropna(axis=0)

df.shape
df.head
df.nunique(axis=0)
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

sns.distplot(df['SalePrice'])
corr = df.corr()

plt.figure(figsize=(19, 15))

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
rs = np.random.RandomState(0)

df = pd.DataFrame(rs.rand(10, 10))

corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')