# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv") # Loaded training data

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
gender_rate = train_data[["Sex","Survived"]].groupby(['Sex'], as_index = False).mean()

print(gender_rate)
pclass_dependence = train_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index = False).mean()

print(pclass_dependence)
SibSp_dependence =  train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index = False).mean()

print(SibSp_dependence)
ParCh_dependence =  train_data[["Parch", "Survived"]].groupby(['Parch'], as_index = False).mean()

print(ParCh_dependence)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g1 = sns.FacetGrid(train_data, col = 'Survived', row = 'Pclass', height = 2.2, aspect = 1.6)

g1.map(plt.hist, 'Age', alpha = 0.8, bins = 20) 
g2 = sns.FacetGrid(train_data, row = 'Embarked', height = 2.2, aspect = 1.6)

g2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')

g2.add_legend()
g3 = sns.FacetGrid(train_data, row = 'Embarked', col = 'Survived', height = 2.2, aspect = 1.6)

g3.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = None)

g3.add_legend()
train_data.head()
# Cabin number and ticket number have too many discrepancies i.e. missing and redundant data. Also there is no intuitive conclusion from these.

# Hence we drop these features. To maintain consistency we will drop them from both the training and test set.

# Also as SEC(Socio-Economic Class) is already signified by the Pclass we'll not be analyzing titles as part of the passenger names

train_data = train_data.drop(['Ticket','Cabin'], axis = 1)

test_data = test_data.drop(['Ticket','Cabin'], axis = 1)

train_data.head()
# converting Sex into numerical values

combine = [train_data, test_data]

for dataset in combine:

     # changing male/female to 0/1 respectively, map takes dictionary as its parameter

    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)



train_data.head()
grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2, 3))

# make the array that will store the predicted values for each value of Sex and Pclass
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            # select those rows with the certain value of Sex, Pclass. Then select only the Age column with the null values dropped

            age_guess = guess_df.median()

            # get the median value from all the data

            guess_ages[i, j] = int(age_guess/0.5 + 0.5) * 0.5

            # Convert random age float to nearest .5 age

    # Now that we have made the guess matrix, we need to put the appropriate values from the matrix into the column of age wherever the values are missing.

    

    for i in range(0, 2):

        for j in range(0, 3):

            # Finding the location of missing values and putting in the calculated average

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

    

train_data.head()
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index = False).mean().sort_values(by = 'AgeBand')

# Cut the total range of age into 5 bands

# We want the columns AgeBand and corresponding survival which has been grouped by the AgeBand and it isn't the index
for dataset in combine:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ (dataset['Age'] > 64), 'Age']
train_data = train_data.drop(['AgeBand'], axis = 1)

combine = [train_data, test_data]

train_data.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean().sort_values(by='Survived', ascending = False)
train_data = train_data.drop(['Parch', 'SibSp'], axis=1)

test_data = test_data.drop(['Parch', 'SibSp'], axis=1)

combine = [train_data, test_data]



train_data.head()
freq_port = train_data.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_data.head()
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

# We will now create the FareBand



train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_data = train_data.drop(['FareBand'], axis=1)

combine = [train_data, test_data]

    

train_data.head(10)
# Firstly we'll change FamilySize to IsAlone, whether the passenger is travelling alone or not

for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# We can now drop FamilySize

train_data = train_data.drop(['FamilySize'], axis=1)

test_data = test_data.drop(['FamilySize'], axis=1)

combine = [train_data, test_data]



train_data.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])



for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_data.head()



train_data = train_data.drop(['Name', 'PassengerId'], axis=1)

test_data = test_data.drop(['Name'], axis=1)

combine = [train_data, test_data]
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)