# This is my first kernel to play around not much to see here if you've seen any kernel ever



import numpy as np

import pandas as pd



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
train_df.describe()
train_df.info()
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
mean_age = np.mean(train_df['Age'])

nan_age = np.isnan(train_df['Age'])

train_df.loc[nan_age, 'Age'] = mean_age
train_df.head()
train_df['Embarked'].value_counts()
train_df['Sex'].value_counts()
train_df['port'] = train_df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).fillna(0)

train_df['female'] = train_df['Sex'].map({'female': 1, 'male': 0})

train_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)

train_df.info()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



sns.pairplot(train_df, hue='Pclass', diag_kind='')

#diag_kind messes up results
train_df.drop('Fare', axis=1, inplace=True)

train_df.head()
sns.heatmap(train_df.corr(), annot=True)
pass_ids = test_df['PassengerId']

test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1, inplace=True)

test_df['port'] = test_df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).fillna(0)

test_df['female'] = test_df['Sex'].map({'female': 1, 'male': 0})

test_df.drop(['Sex', 'Embarked'], axis=1, inplace=True)

nan_age_test = np.isnan(test_df['Age'])

test_df.loc[nan_age_test, 'Age'] = mean_age

test_df.head()
test_df.info()
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

model.fit(train_df.drop('Survived', axis=1), train_df['Survived'])

y_pred = model.predict(test_df)

prediction = pd.DataFrame({'PassengerId': pass_ids, 'Survived': y_pred})

prediction.head()
prediction.to_csv('submission.csv', index = False)