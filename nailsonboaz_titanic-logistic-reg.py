import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression 
# Read files

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

# Train.csv will contain the details of a subset of the passengers on board (891 to be exact) and importantly, the “ground truth”.

train = pd.read_csv("../input/titanic/train.csv")

# The test.csv dataset contains similar information but does not disclose the “ground truth” for each passenger. It’s your job to predict these outcomes.

test = pd.read_csv("../input/titanic/test.csv")

train.head()
train.info()
fig, ax = plt.subplots(1,4, figsize=(15,3))



train.Sex.value_counts().plot(kind='bar', ax=ax[0])

train.Embarked.value_counts().plot(kind='bar', ax=ax[1])

train.Pclass.value_counts().plot(kind='bar', ax=ax[2])



df_plot = train.groupby(['Pclass', 'Embarked']).size().reset_index().pivot(columns='Pclass', index='Embarked', values=0)

df_plot.apply(lambda x: x/x.sum(), axis=1).plot(kind='bar', stacked=True, ax=ax[3])
train.drop(['PassengerId'], axis=1).hist(figsize=(20,10))
pclass_dummies = pd.get_dummies(train.Pclass, prefix = 'Pcclass')

embarked_dummies = pd.get_dummies(train.Embarked, prefix = 'Embarked')

sex_dummies = pd.get_dummies(train.Sex, prefix = 'Sex')



train_y = train['Survived']



titanic_train = train.drop(['PassengerId','Name','Ticket','Cabin','Pclass','Embarked','Sex','Survived'], axis=1)

titanic_train["Age"].fillna(train["Age"].mean(), inplace = True)

titanic_train["Fare"].fillna(train["Fare"].mean(), inplace = True)



titanic_train = pd.concat([titanic_train,pclass_dummies,embarked_dummies,sex_dummies], axis=1)

titanic_train.head()
titanic_train[titanic_train.isna().any(axis=1)]
reg = LogisticRegression().fit(titanic_train, train_y)

reg.score(titanic_train, train_y)
pclass_dummies = pd.get_dummies(test.Pclass, prefix = 'Pcclass')

embarked_dummies = pd.get_dummies(test.Embarked, prefix = 'Embarked')

sex_dummies = pd.get_dummies(test.Sex, prefix = 'Sex')



titanic_test = test.drop(['PassengerId','Name','Ticket','Cabin','Pclass','Embarked','Sex'], axis=1)

titanic_test["Age"].fillna(train["Age"].mean(), inplace = True)

titanic_test["Fare"].fillna(train["Fare"].mean(), inplace = True)



titanic_test = pd.concat([titanic_test,pclass_dummies,embarked_dummies,sex_dummies], axis=1)

titanic_test
titanic_test[titanic_test.isna().any(axis=1)]
pred = reg.predict(titanic_test)



my_submission = test[['PassengerId']]

my_submission['Survived'] = pred

my_submission.head()
my_submission.to_csv('my_submission.csv', index=False)