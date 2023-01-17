# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.DataFrame(pd.read_csv('../input/indian_liver_patient.csv'))
data.head()
data.info()
data.sample(5)
data.shape
data.describe()
missing_values = data.isnull().sum()

missing_values
data=data.dropna()

data.shape
print('Number of people in Dataset 1',data[data['Dataset'] == 1].Age.count())

print('Number of people in Dataset 2',data[data['Dataset'] == 2].Age.count())
sns.countplot(x='Dataset',data=data)

plt.show()
# Gender Distribution of 2 Dataset

sns.countplot(x='Gender',data=data,hue='Dataset',palette="Set1")

plt.title('Distribution of Datasets by Gender')

plt.show()
columns=list(data.columns[:10])

columns.remove('Gender')

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    data[i].hist(bins=10,edgecolor='black')#,range=(0,0.3))

    plt.title(i)

plt.show()
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

plt.subplots_adjust(wspace=0.2,hspace=0.5)

data[data['Gender']=='Male'].Age.plot(ax=ax1, kind='hist', bins=10,edgecolor='black')

ax1.set_title('Male Distribution')

data[data['Gender']=='Female'].Age.plot(ax=ax2, kind='hist',bins=10,edgecolor='black')

ax2.set_title('Female Distribution')

plt.show()
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

plt.subplots_adjust(wspace=0.2,hspace=0.5)

data[(data['Gender']=='Male') & (data['Dataset'] == 1)].Age.plot(ax=ax1, kind='hist', bins=10,edgecolor='black')

ax1.set_title('Male Distribution')

data[(data['Gender']=='Female') & (data['Dataset'] == 1)].Age.plot(ax=ax2, kind='hist',bins=10,edgecolor='black')

ax2.set_title('Female Distribution')

plt.show()
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)

plt.subplots_adjust(wspace=0.2,hspace=0.5)

data[(data['Gender']=='Male') & (data['Dataset'] == 2)].Age.plot(subplots=True,ax=ax1, kind='hist', bins=10,edgecolor='black')

ax1.set_title('Male Distribution')

data[(data['Gender']=='Female') & (data['Dataset'] == 2)].Age.plot(subplots=True,ax=ax2, kind='hist',bins=10,edgecolor='black')

ax2.set_title('Female Distribution')

plt.show()