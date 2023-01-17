import pandas as pd

import numpy as np
train = pd.read_csv('../input/train.csv')
train.columns.tolist()
train.info()
train.head(n=10)
columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare']

train.drop(columns, inplace=True, axis=1)
columns = ['Pclass', 'Sex', 'Embarked']

for column in columns:

    train[column] = train[column].astype('category')
train['Survived'] = train['Survived'].astype('bool')
import matplotlib.pyplot as plt

import seaborn as sns
train.head()
train.describe()
train['Survived'].mean()
train.groupby('Pclass')['Survived'].mean()
nummale = train[train['Sex']=='male']['Sex'].count()

numfemale = len(train)-nummale

print('There are', nummale,'male passengers and', numfemale, 'female passengers')
train.groupby('Sex')['Survived'].mean()
sns.set()

train.hist(column='Age')

plt.xlabel('Age')

plt.title('')

plt.show()
train.hist(column='Age', by='Survived', sharey=True)

plt.show()
train.hist(column='Age', by='Sex', sharey=True)

plt.show()
train.hist(column='Age', by='Pclass', sharey=True)

plt.show()
train.hist(column='SibSp')

plt.show()
train.hist(column='Parch')

plt.show()
train['withsibsp'] = train['SibSp'] > 0

train['withparch'] = train['Parch'] > 0
train.groupby('withsibsp')['Survived'].mean()
train.groupby('withparch')['Survived'].mean()
train.groupby('Embarked')['Embarked'].count()
train.groupby('Embarked')['Survived'].mean()