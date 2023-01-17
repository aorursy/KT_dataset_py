# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Imports

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt
# Read 'train' and 'test' data

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
print(train.shape)

train.head()
print(test.shape)

test.head()
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
#1

train.Sex.value_counts()/len(train)
sns.barplot(x='Sex', y='Survived',data=train)

plt.show()
#2

train.Embarked.value_counts()/len(train)
sns.lineplot(x='Embarked', y='Survived', data=train)

plt.show()
#3

sns.FacetGrid(train, hue='Survived', height=8).map(sns.distplot,'Age').set_axis_labels('Age','Survived').add_legend()

plt.show()
#4

sns.barplot(x='Pclass',y='Survived' ,data=train)

plt.show()