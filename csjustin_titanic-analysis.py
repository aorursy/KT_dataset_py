# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
train = pd.read_csv('../input/train.csv')
train.head()
sns.countplot(x='Survived', data=train, hue='Sex')

plt.legend(bbox_to_anchor=(1.05,1), loc=2)
sns.countplot(x='Survived', data=train, hue='Pclass')

plt.legend(bbox_to_anchor=(1.05,1), loc=2)
sns.countplot(x='Survived', data=train, hue='SibSp')

plt.legend(bbox_to_anchor=(1.05,1), loc=2)
sns.distplot(train['Age'].dropna(), bins=35)
sns.distplot(train['Parch'], kde=False)
sns.distplot(train['Fare'], kde=False)
sns.boxplot(x='Pclass', y='Age', data=train)

plt.figure(figsize=(10,4))
def impute_age(col):

    age = col[0]

    pclass = col[1]

    

    if pd.isnull(age):

        if pclass == 1:

            return train[train['Pclass'] == 1]['Age'].mean()

        if pclass == 2:

            return train[train['Pclass'] == 2]['Age'].mean()

        if pclass == 3:

            return train[train['Pclass'] == 3]['Age'].mean()

    else:

        return age
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
train.head()
Pclass = pd.get_dummies(train['Pclass'], drop_first=True)

Sex = pd.get_dummies(train['Sex'], drop_first=True)

Embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, Sex, Pclass, Embarked], axis = 1)     
train.head()
train.drop(['PassengerId', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace=True)
train.head()
from sklearn.linear_model import LogisticRegression
X = train.drop('Survived', axis=1)

y = train['Survived']
logmodel = LogisticRegression()
logmodel.fit(X,y)
test = pd.read_csv('../input/test.csv')
test.head()
Pclass_test = pd.get_dummies(test['Pclass'], drop_first=True)

Sex_test = pd.get_dummies(test['Sex'], drop_first=True)

Embarked_test = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test, Sex_test, Pclass_test, Embarked_test], axis = 1) 
def impute_age_test(col):

    age = col[0]

    pclass = col[1]

    

    if pd.isnull(age):

        if pclass == 1:

            return test[test['Pclass'] == 1]['Age'].mean()

        if pclass == 2:

            return test[test['Pclass'] == 2]['Age'].mean()

        if pclass == 3:

            return test[test['Pclass'] == 3]['Age'].mean()

    else:

        return age
test['Age'] = test[['Age', 'Pclass']].apply(impute_age_test, axis=1)
test.drop('Cabin', axis=1, inplace=True)

test.dropna(inplace=True)
