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
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/Interview.csv')
data = df.copy()
data.shape
data.head()
data.describe()
data.drop(data.columns[-5:], axis=1, inplace=True)
for i,j in enumerate(data.columns):
    print(i,j)
pd.set_option('display.max_columns', 30)
data.describe()
for idx, feat in enumerate(data.columns):
    print(idx, feat, end="\t\t")
    print(data[feat].isnull().sum())
data.isnull().sum(axis=1).unique()
data[data.isnull().sum(axis=1)==22]
data.drop(1233, inplace=True)
for feat in data.drop('Name(Cand ID)', axis=1).columns:
    plt.figure(figsize=(15, 5))
    sns.countplot(x=feat, data=data.drop('Name(Cand ID)', axis=1))
    if data[feat].unique().size > 8:
        plt.xticks(rotation=45, ha='right')
    plt.show()
data[data.columns[-2]].unique()
data.fillna('na', inplace=True)
for feat in data.columns:
    if type(data[feat][0]) == str:
        data[feat] = data[feat].map(lambda x: x.lower())

