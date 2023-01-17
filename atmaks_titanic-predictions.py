import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import pandas
train_df = pandas.read_csv("../input/train.csv")

test_df = pandas.read_csv("../input/test.csv")

train_df.info()
train_df.head(5)
train_df.describe(include=['O'])
sns.countplot(x='Survived', data=train_df)
g=sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist,'Age', bins =20)
g = sns.FacetGrid(train_df, col='Survived',row='Pclass')

g.map(plt.hist,'Age',bins=10)
train_df = train_df.drop(['Name','Ticket','Embarked'], axis = 1)

test_df = test_df.drop(['Name','Ticket','Embarked'], axis =1)

train_df.head()
train_df['Sex'] = train_df['Sex'].replace({'male' : 1, 'female' : 0})

test_df['Sex'] = test_df['Sex'].replace({'male' : 1, 'female' : 0})
from sklearn import preprocessing

fields = ['Sex',]

train_df.groupby('Parch')['Parch'].count()
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())

train_df.info()
del train_df['Cabin']

del test_df['Cabin']
train_df.head()
train_data = train_df.values

X_train = train_data[:,2:]

y_train = train_data[:,1]

print(X_train.shape, y_train.shape)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())

test_df.info()

test_df.head()
test_data = test_df.values

X_test = test_data[:,1:]

predictions = rf.predict(X_test)

output = pandas.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived': predictions})

output.set_index('PassengerId',inplace=True)

output.to_csv('output.csv', header=True)



                           