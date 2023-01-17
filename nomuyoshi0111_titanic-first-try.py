# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



train_df.head(3)
print("----- training data info -----")

train_df.info()

print("----- test data info -----")

test_df.info()
# Drop unuseful columns.

# Cabin column has a lot of NaN values.

train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) # when drop column, axis=1

test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# plot 'Embarked'

fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15, 5))

sns.countplot(x='Embarked', data=train_df, ax=axis1)

sns.countplot(x='Survived', hue='Embarked', data=train_df, ax=axis2)

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
# Drop Embarked column, because Embarked does not seem to be useful in prediction.

train_df = train_df.drop(['Embarked'], axis=1)

test_df = test_df.drop(['Embarked'], axis=1)
train_df.describe()
sns.countplot(x='Survived', hue='Sex', data=train_df)
# Females are survived more than males.

# I think that it is useful in prediction.

# Replace Male: 1, Female: 0

train_df = train_df.replace({'Sex': {'male': 1, 'female': 0}})

test_df = test_df.replace({'Sex': {'male': 1, 'female': 0}})
# Replace null values to median value.

age_median = train_df['Age'].median()

train_df.fillna({'Age': age_median}, inplace=True)

test_df.fillna({'Age': age_median}, inplace=True)



fare_median_test = test_df['Fare'].median()

test_df.fillna({'Fare': fare_median_test}, inplace=True)
fig, ax = plt.subplots(1, 1, figsize=(20, 10))

plt.xlim(0, 90)

plt.xticks(np.arange(0, 90, 2))

sns.distplot(train_df[train_df['Survived']==1]['Age'], color='r', ax=ax, hist=False, label='Survived')

sns.distplot(train_df[train_df['Survived']==0]['Age'], ax=ax, hist=False, label='Not Survived')
fig, ax = plt.subplots(1, 1, figsize=(20, 10))

plt.xlim(0, 90)

plt.xticks(np.arange(0, 90, 2))

sns.distplot(train_df[(train_df['Survived']==1) & (train_df['Sex']==0)]['Age'], color='r', ax=ax, hist=False, label='Survived & Female')

sns.distplot(train_df[(train_df['Survived']==0) & (train_df['Sex']==0)]['Age'], color='m', ax=ax, hist=False, label='Not Survived & Female')

sns.distplot(train_df[(train_df['Survived']==1) & (train_df['Sex']==1)]['Age'], color='b', ax=ax, hist=False, label='Survived & Male')

sns.distplot(train_df[(train_df['Survived']==0) & (train_df['Sex']==1)]['Age'], color='c', ax=ax, hist=False, label='Not Survived & Male')
# create new features

# Child and Child*Sex

# train

train_df['Child'] = train_df['Age'] < 17

train_df['Child'] = train_df['Child'] * 1

train_df['Child*Sex'] = train_df['Child'] * train_df['Sex']



# test

test_df['Child'] = test_df['Age'] < 17

test_df['Child'] = test_df['Child'] * 1

test_df['Child*Sex'] = test_df['Child'] * test_df['Sex']
#from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
# create training_set, test_set

X_train = train_df.drop(['Survived'], axis=1)

X_test = test_df.drop(['PassengerId'], axis=1)

y_train = train_df['Survived']
logistic_model = LogisticRegression(penalty='l1')

logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

train_score = logistic_model.score(X_train, y_train)

print("logistic y_pred: {}".format(y_pred))

print("logistic train_score: {}".format(train_score))
logistic_model.coef_
submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": y_pred

})

submission.to_csv("titanic.csv", index=False)