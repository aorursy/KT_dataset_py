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
gender_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.columns
train_data.head()
train_data.info()
train_data.describe()
train_data.describe(include = ['O'])
import seaborn as sns



sns.barplot(y=train_data['Survived'], x=train_data['Pclass'])
sns.jointplot(x=train_data['Age'], y=train_data['Survived'], kind="kde")
sns.barplot(x=train_data['Sex'], y=train_data['Survived'])
sns.lineplot(x=train_data['Parch'], y=train_data['Survived'])
sns.lineplot(x=train_data['SibSp'], y=train_data['Survived'])
train_data[train_data.columns[1:]].corr()['Survived'][:]
import matplotlib.pyplot as plt

g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=2)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_data, row='Embarked', size=3.3, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=3.3, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train_data = train_data.drop(['Ticket', 'Cabin'], axis = 1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis = 1)
df = [train_data, test_data]

for row in df:

    row['Title'] = row.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])
for row in df:

    row['Title'] = row['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    row['Title'] = row['Title'].replace('Mlle', 'Miss')

    row['Title'] = row['Title'].replace('Ms', 'Miss')

    row['Title'] = row['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for row in df:

    row['Title'] = row['Title'].map(title_mapping)

    row['Title'] = row['Title'].fillna(0)
train_data = train_data.drop(['Name', 'PassengerId'], axis=1)

test_data = test_data.drop(['Name'], axis=1)
df = [train_data, test_data]
train_data.head()
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()



train_data['Sex'] = encoder.fit_transform(train_data['Sex'])

test_data['Sex'] = encoder.fit_transform(test_data['Sex'])
train_data.head()
test_data.head()
grid = sns.FacetGrid(train_data, col='Pclass', row='Sex', size=3.3, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))

df = [train_data, test_data]

guess_ages

for row in df:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = row[(row['Sex'] == i) & \

                                  (row['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            row.loc[ (row.Age.isnull()) & (row.Sex == i) & (row.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    row['Age'] = row['Age'].astype(int)



train_data.head()
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data.head()
train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_data['AgeBand'] = encoder.fit_transform(train_data['AgeBand'])

train_data.head()
test_data['AgeBand'] = pd.cut(test_data['Age'], 5)

test_data['AgeBand'] = encoder.fit_transform(test_data['AgeBand'])

test_data.head()
df = [train_data, test_data]

for row in df:

    row['FamilySize'] = row['SibSp'] + row['Parch'] + 1



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for row in df:

    row['IsAlone'] = 0

    row.loc[row['FamilySize'] == 1, 'IsAlone'] = 1



train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_data = train_data.drop(['Parch', 'SibSp'], axis = 1)

test_data = test_data.drop(['Parch', 'SibSp'], axis = 1)
test_data.head()
train_data.head()
train_data['Age*Class'] = train_data['AgeBand'] * train_data['Pclass']

train_data.head()

test_data['Age*Class'] = test_data['AgeBand'] * test_data['Pclass']

train_data.head()
freq_port = train_data.Embarked.dropna().mode()[0]

freq_port
train_data['Embarked'] = train_data['Embarked'].fillna(freq_port)

test_data['Embarked'] = test_data['Embarked'].fillna(freq_port)
train_data.head()
train_data['Embarked'] = encoder.fit_transform(train_data['Embarked'])

train_data.head()
test_data['Embarked'] = encoder.fit_transform(test_data['Embarked'])

test_data.head()

df = [train_data, test_data]
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

train_data.head()
freq_fare = test_data.Fare.dropna().mode()[0]

freq_fare.astype('int')

test_data['Fare'] = test_data['Fare'].fillna(freq_fare)

test_data.info()
for row in df:

    row.loc[ row['Fare'] <= 7.91, 'Fare'] = 0

    row.loc[(row['Fare'] > 7.91) & (row['Fare'] <= 14.454), 'Fare'] = 1

    row.loc[(row['Fare'] > 14.454) & (row['Fare'] <= 31), 'Fare']   = 2

    row.loc[ row['Fare'] > 31, 'Fare'] = 3

    row['Fare'] = row['Fare'].astype(int)



train_data = train_data.drop(['FareBand'], axis=1)

combine = [train_data, test_data]

    

train_data.head(10)
test_data.head()
train_data = train_data.drop(['Age', 'FamilySize'], axis = 1)

test_data = test_data.drop(['Age', 'FamilySize'], axis = 1)
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred

    })
submission.to_csv('submission.csv', index=False)