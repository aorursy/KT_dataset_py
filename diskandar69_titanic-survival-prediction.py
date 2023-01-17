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
# data analysis

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
train_df.columns
test_df.columns
train_df.head(5)
test_df.head(5)
train_df.info()
test_df.info()
train_df.describe().T
test_df.describe().T
train_df.shape
test_df.shape
train_df.isna().sum()
test_df.isna().sum()
train_df.head(2)
plt.figure(figsize=(15,5))

sns.barplot(x="Parch", y="Survived", hue="Sex", data=train_df)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_df)

plt.show()
plt.figure(figsize=(15,5))

sns.barplot(x="SibSp", y="Survived", hue="Sex", data=train_df)

plt.show()
train_df["SibSp"].value_counts()
f,ax = plt.subplots(3,4,figsize=(20,16))

sns.countplot('Pclass',data=train_df,ax=ax[0,0])

sns.countplot('Sex',data=train_df,ax=ax[0,1])

sns.boxplot(x='Pclass',y='Age',data=train_df,ax=ax[0,2])

sns.countplot('SibSp',hue='Survived',data=train_df,ax=ax[0,3],palette='husl')

sns.distplot(train_df['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')

sns.countplot('Embarked',data=train_df,ax=ax[2,2])



sns.countplot('Pclass',hue='Survived',data=train_df,ax=ax[1,0],palette='husl')

sns.countplot('Sex',hue='Survived',data=train_df,ax=ax[1,1],palette='husl')

sns.distplot(train_df[train_df['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)

sns.distplot(train_df[train_df['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)

sns.countplot('Parch',hue='Survived',data=train_df,ax=ax[1,3],palette='husl')

sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train_df,palette='husl',ax=ax[2,1])

sns.countplot('Embarked',hue='Survived',data=train_df,ax=ax[2,3],palette='husl')



ax[0,0].set_title('Total Passengers by Class')

ax[0,1].set_title('Total Passengers by Gender')

ax[0,2].set_title('Age Box Plot By Class')

ax[0,3].set_title('Survival Rate by SibSp')

ax[1,0].set_title('Survival Rate by Class')

ax[1,1].set_title('Survival Rate by Gender')

ax[1,2].set_title('Survival Rate by Age')

ax[1,3].set_title('Survival Rate by Parch')

ax[2,0].set_title('Fare Distribution')

ax[2,1].set_title('Survival Rate by Fare and Pclass')

ax[2,2].set_title('Total Passengers by Embarked')

ax[2,3].set_title('Survival Rate by Embarked')
train_df
train_df = train_df.drop(['PassengerId'], axis=1)
# we can now drop the cabin feature

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
train_df["Age"].isna().sum()
train_df.Age = train_df.Age.fillna(train_df.Age.mean())

train_df
train_df['Embarked'].describe()
common_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_df.info()
data = [train_df, test_df]

for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset["Fare"].astype(int)
train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)
genders = {"male":0, "female":1}

data = [train_df, test_df]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_df.head(2)
train_df['Ticket'].describe()
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
train_df["Embarked"].value_counts()
port = {"S":0, "C":1, "Q":2}

data = [train_df, test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(port)
test_df.fillna(test_df.mean(), inplace=True)
test_df.isna().sum()
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(round(acc_log,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)



Y_pred = gaussian.predict(X_test)



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print(round(acc_gaussian,2,), "%")
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)



Y_pred = decision_tree.predict(X_test)



acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(round(acc_decision_tree,2,), "%")
results = pd.DataFrame({

    'Model': ['Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Decision Tree'],

    'Score': [acc_log, acc_random_forest, acc_gaussian,

              acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_prediction

    })

submission.to_csv('submission.csv', index=False)