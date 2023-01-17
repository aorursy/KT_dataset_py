# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import math



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()



def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))



bar_chart('Sex')
train['Sex'] = train['Sex'].map({'male':0,'female':1})

train.loc[:,'Age'] = np.floor(train['Age']/10)*10

train.loc[:,'SibSp']=train['SibSp']-train['Parch']

train.loc[(train['Fare'] < 40),:'Fare'] = 0.2

train.loc[(train['Fare'] >= 40) & (train['Fare'] < 80),:'Fare'] = 0.4

train.loc[(train['Fare'] >= 80) & (train['Fare'] < 120),:'Fare'] = 0.6

train.loc[(train['Fare'] >= 120) & (train['Fare'] < 160),:'Fare'] = 0.8

train.loc[(train['Fare'] >= 160),:'Fare'] = 1

train.drop('PassengerId',axis=1,inplace=True)

train.drop('Ticket',axis=1,inplace=True)

train.drop('SibSp',axis=1,inplace=True)

train.drop('Embarked',axis=1,inplace=True)

train.drop('Cabin',axis=1,inplace=True)



test['Sex'] = test['Sex'].map({'male':0,'female':1})

test.loc[:,'Age'] = np.floor(test['Age']/10)*10

test.loc[:,'SibSp']=test['SibSp']-test['Parch']

test.loc[(test['Fare'] < 40),:'Fare'] = 0.2

test.loc[(test['Fare'] >= 40) & (test['Fare'] < 80),:'Fare'] = 0.4

test.loc[(test['Fare'] >= 80) & (test['Fare'] < 120),:'Fare'] = 0.6

test.loc[(test['Fare'] >= 120) & (test['Fare'] < 160),:'Fare'] = 0.8

test.loc[(test['Fare'] >= 160),:'Fare'] = 1

test.drop('PassengerId',axis=1,inplace=True)

test.drop('Ticket',axis=1,inplace=True)

test.drop('SibSp',axis=1,inplace=True)

test.drop('Embarked',axis=1,inplace=True)

test.drop('Cabin',axis=1,inplace=True)





target = train['Survived']
from sklearn.svm import SVC

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv=k_fold, n_jobs=1, scoring=scoring)