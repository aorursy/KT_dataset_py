# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir(".."))



# Any results you write to the current directory are saved as output.
train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')
train_set.head(10)
print(train_set.columns)
# How many survived in this sample?

sum(train_set['Survived']/len(train_set['Survived']))
train_set.info()
test_set.info()
train_set.describe()
# include objects only in summary

train_set.describe(include=['O'])
# See how much Pclass affects survival:

train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived')
import seaborn as sns

import matplotlib.pyplot as plt

g = sns.FacetGrid(train_set, col='Survived')

g.map(plt.hist, 'Age', bins=40)
grid = sns.FacetGrid(train_set, col='Pclass', hue='Survived')

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()

h = sns.FacetGrid(train_set, col='Survived', hue='Sex')

h.map(plt.hist, 'Age', alpha=0.6, bins=20)

h.add_legend()
g = sns.FacetGrid(train_set, row='Embarked', aspect=1.6)

g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

g.add_legend()
grid = sns.FacetGrid(train_set, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
# Drop Features

train_set = train_set.drop(['Cabin', 'Ticket'], axis=1)

test_set = test_set.drop(['Cabin', 'Ticket'], axis=1)
# Create Features

train_set['Title'] = train_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_set['Title'] = test_set.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_set.head()
pd.crosstab(train_set['Title'], train_set['Sex'])
combine=[train_set,test_set]

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train_set['Title'] = train_set['Title'].map(title_mapping)

train_set['Title'] = train_set['Title'].fillna(0)

test_set['Title'] = test_set['Title'].map(title_mapping)

test_set['Title'] = test_set['Title'].fillna(0)



train_set.head()
train_set = train_set.drop(['Name', 'PassengerId'], axis=1)

test_set = test_set.drop(['Name', 'PassengerId'], axis=1)

train_set.shape
combine = [train_set, test_set]

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_set.head()
import numpy as np

guess_ages = np.zeros((2,3))
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j].astype(int)



    dataset['Age'] = dataset['Age']



train_set.head(20)
train_set['AgeBand'] = pd.cut(train_set['Age'], 5)

train_set[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_set.head()
train_set = train_set.drop(['AgeBand'], axis=1)

combine = [train_set, test_set]

train_set.head()
combine = [train_set, test_set]

for dataset in combine:

    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1



train_set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived')
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train_set[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived')
train_set = train_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_set = test_set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_set, test_set]

train_set.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_set.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_set.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_set.head()
test_set['Fare'].fillna(test_set['Fare'].dropna().median(), inplace=True)

test_set.head()
train_set['FareBand'] = pd.qcut(train_set['Fare'], 4)

train_set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_set = train_set.drop(['FareBand'], axis=1)

combine = [train_set, test_set]

    

train_set.head(10)
test_set.head()
X_train = train_set.drop("Survived", axis=1)

Y_train = train_set["Survived"]

X_test  = test_set.copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

logreg= LogisticRegression()

logreg.fit(X_train, Y_train)

logreg.predict(X_test)

score_logreg = logreg.score(X_train, Y_train) * 100

score_logreg
# Remove 'Survived column'

coeff_df = pd.DataFrame(train_set.columns.delete(0))

coeff_df.rename(columns={0: 'Features'}, inplace=True)
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation')
new_X_train = X_train.drop('IsAlone', axis=1)

new_X_test = X_test.drop('IsAlone', axis=1)
logreg.fit(new_X_train, Y_train)

logreg.predict(new_X_test)

logreg.score(new_X_train, Y_train) * 100
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train, Y_train)

svc.predict(X_test)

svc.score(X_train, Y_train) * 100
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

knn.predict(X_test)

knn.score(X_train, Y_train) * 100 
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, Y_train)

rf.predict(X_test)

rf.score(X_train, Y_train) * 100


