# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/oasis_cross-sectional.csv')
dataset
dataset.head()
dataset.describe()
dataset.columns
dataset2=dataset

                        

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, dataset2.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(dataset2.columns.values[i - 1])

   

    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values

    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])



print(dataset['ID'])
dataset['M/F']


dataset["Age"].median()
dataset["Age"].mean()
dataset3=pd.read_csv("../input/oasis_longitudinal.csv")
dataset3
dataset3.describe()
dataset3["Age"].min()
dataset3["Age"].max()
dataset3["Age"].mean()
dataset3[dataset3.Group=='Demented'].count()
dataset3[dataset3.Group=='Nondemented'].count()
dataset3[dataset3['M/F']=="F"].count()
dataset3[dataset3['M/F']=="M"].count()
dataset3[dataset3['Hand']=="R"].count()
dataset3[dataset3['Hand']=="L"].count()
dataset3['Hand']
fig = plt.figure(figsize=(15, 12))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, dataset3.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(dataset3.columns.values[i - 1])

   

    values = dataset3.iloc[:, i - 1].value_counts(normalize = True).values

    index = dataset3.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
