# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import networkx as nx



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Show versions

print('Numpy Version:',np.__version__)

print('Pandas Version:',pd.__version__)

print('Matplotlib Version:',plt.__version__)

print('Networkx Version:',nx.__version__)
df = pd.read_csv('../input/titanicdataset-traincsv/train.csv')
df.head()
df.shape
df.tail()
df.describe
df.info()
df.isnull()
df.count()
columns = df.columns

index = df.index

data = df.to_numpy()

print(columns)

print(index)

print(data)
index.to_numpy()
columns.to_numpy()
df.dtypes
df.dtypes.value_counts()
df["Age"]
df.loc[:, "Name"]
df.iloc[:,1]
df.min()
df.max()
df.mean()
df.median()