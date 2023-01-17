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
import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
train_df.info()
train_df['Survived'].value_counts()
sns.countplot(x=('Survived'),hue='Pclass',data=train_df)
sns.countplot(x=train_df['Survived'],hue=train_df['Pclass'])
train_df['Survived'].groupby(train_df['Pclass']).mean()
train_df['Name'].head()
train_df['Name_title'] = train_df['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x: x.split()[0])

train_df['Name_title'].value_counts()
train_df['Survived'].groupby(train_df['Name_title']).mean()
train_df['Sex'].head()
sns.countplot(x=train_df['Survived'],hue=train_df['Sex'])
train_df['Age'].describe()
plt.hist(train_df['Age'],bins=5)
train_df['Age_cat'] = train_df['Age']/20
train_df.Age_cat.astype('int64')