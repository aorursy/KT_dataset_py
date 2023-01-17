# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#Helpers
def single_countplot(x=None, data=None, hue=None, xlabel='Xlabel', ylabel='Ylabel', title='Title', fig_size=(15,7)):
    """
    Draws a single count plot
    """
    sns.set(rc={'figure.figsize':fig_size})
    ax = sns.countplot(x=x, data=data, hue=hue)
    ax.set_title(title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax
os.listdir("../input/train")
raw_train = pd.read_csv("../input/train/train.csv")
raw_test = pd.read_csv("../input/test/test.csv")
raw_train.head(3)
train = raw_train.drop(columns='Description', axis=1)
test = raw_test.drop(columns='Description', axis=1)
train['DatasetType'] = 'Train'
test['DatasetType'] = 'Test'
data = pd.concat([train, test], sort=True)
print(train.shape)
train.info()
train.describe()
data['Type'] = data['Type'].apply(lambda x: 'Dog' if x == 1 else 'Cat')
single_countplot(x='DatasetType',
                data=data,
                hue='Type',
                xlabel='Dataset Type',
                ylabel='Pet count',
                title='Counts of Dogs vs Cats across Train/Test Sets')    

#TODO: add percentage comparison: Type Count vs Total count in the Adoption Speed Group / Type Count vs Total Count of Type
single_countplot(x='AdoptionSpeed',
                data=data[data.DatasetType == 'Train'],
                hue='Type',
                xlabel='Adoption Speed Range',
                ylabel='Pet count',
                title='Counts of Dogs vs Cats by Adoption Speed Ranges')
for speed in sorted(data[data.DatasetType == 'Train'].AdoptionSpeed.unique()):
    axes = plt.hist(data[(data.DatasetType == 'Train') & 
                       (data.AdoptionSpeed == speed)].Age,
                       label='Adoption speed {0}'.format(str(speed)),
                       alpha=0.3,
                       bins=30)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age across Adoption Speed Ranges in Train Set')
ax = plt.hist(data[(data.DatasetType == 'Test')].Age,
                   alpha=0.3,
                   bins=30)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age across Adoption Speed Ranges in Test Set')
fig, axes = plt.subplots(1,2)
axes[0].scatter(data[data.DatasetType == 'Train'].Age, data[data.DatasetType == 'Train'].Fee)
axes[0].set(xlabel='Age', ylabel='Fee')
axes[0].set_title('Train Set')
axes[1].scatter(data[data.DatasetType == 'Test'].Age, data[data.DatasetType == 'Test'].Fee)
axes[1].set(xlabel='Age', ylabel='Fee')
axes[1].set_title('Test Set')

ax = sns.boxplot(x="AdoptionSpeed", y="Fee", data=data[data.DatasetType == 'Train'], hue='Health')
ax.set_title('Boxplot of Fee values distribution for Adoption Speed Ranges across pets Health condition')
for speed in sorted(data[data.DatasetType == 'Train'].AdoptionSpeed.unique()):
    ax = plt.hist(data[(data.DatasetType == 'Train') & 
                       (data.AdoptionSpeed == speed)].Fee,
                       label='Adoption speed {0}'.format(str(speed)),
                       alpha=0.2,
                       bins=100)
plt.legend()
plt.xlabel('Fee')
plt.ylabel('Frequency')
plt.title('Distribution of Fee values across Adoption Speed ranges')
