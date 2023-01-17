# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data.info()
data.corr()
f, ax = plt.subplots(figsize = (25,25))

sns.heatmap(data.corr(), annot=True, linewidths = 2, fmt =".1f", ax=ax)

plt.show()
data.head()
data.columns
query="Finishing"

sort_data = data.copy()

sort_data.sort_values(query, ascending=False, inplace=True)

names=list(sort_data.Name)

index=[i for i in range(len(sort_data.Name))]

sort_data[query].iloc[0:20].plot(kind="bar", color="b", label=query, linewidth=1, alpha=0.5, grid=True)

plt.legend(loc="upper right")

plt.xticks(index[:20],names[:20])

plt.show()
query="Potential"

sort_data1 = data[data.Age<22].copy()

sort_data1.sort_values(query, ascending=False, inplace=True)

xquery="Overall"

yquery="Potential"

sort_data1.iloc[0:2000].plot(kind="scatter", x=xquery, y=yquery, alpha=0.5, color="blue")

plt.xlabel(xquery)

plt.ylabel(yquery)

plt.show()