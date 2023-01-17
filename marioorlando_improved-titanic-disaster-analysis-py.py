# Library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



# For visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# For machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.decomposition import PCA
# Load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

combine = [train_data, test_data]



train_data.head()
# Distribution of data

train_data.describe(include = 'all')
# Dropping features

print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_data, test_data]



print("After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)
# Extract Title from Name

for data in combine:

    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_data['Title'], train_data['Sex'])
# Group the Title

for data in combine:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Drop Name and ID

train_data = train_data.drop(['Name', 'PassengerId'], axis=1)

test_data = test_data.drop(['Name'], axis=1)

combine = [train_data, test_data]

train_data.shape, test_data.shape
train_data.head(20)
pd.crosstab(train_data['Title'], train_data['Survived'])
# Visualization analysis for correlation between Age and Title

g = sns.FacetGrid(train_data, col='Title')

g.map(plt.hist, 'Age', bins=20)
# Get Dummies of Title and Embarked

train_data = pd.get_dummies(train_data, columns=['Title','Embarked'])

test_data = pd.get_dummies(test_data, columns=['Title','Embarked'])

combine = [train_data, test_data]



train_data.head()
# Combine SibSp and Parch

for data in combine:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Drop Parch and SibSp

train_data = train_data.drop(['Parch', 'SibSp'], axis=1)

test_data = test_data.drop(['Parch', 'SibSp'], axis=1)

combine = [train_data, test_data]



train_data.head()
# Categorize Family Size

for data in combine:

    data['Family'] = "Small"

    data.loc[data['FamilySize'] >= 5, 'Family'] = "Large"



train_data.head()
# Drop FamilySize

train_data = train_data.drop(['FamilySize'], axis=1)

test_data = test_data.drop(['FamilySize'], axis=1)

combine = [train_data, test_data]



train_data.head()
# Get Dummies of Family

train_data = pd.get_dummies(train_data, columns=['Family'])

test_data = pd.get_dummies(test_data, columns=['Family'])

combine = [train_data, test_data]



train_data.head()
# Change Sex to Ordinal

for data in combine:

    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_data.head()
# Predict Age using Median and Correlation between Age, Sex, Pclass

guess_ages = np.zeros((2,3))

for data in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = data[(data['Sex'] == i) & \

                                  (data['Pclass'] == j+1)]['Age'].dropna()



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            data.loc[ (data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    data['Age'] = data['Age'].astype(int)



train_data.isnull().sum()
# Cut the Age (Create AgeBand)

train_data['AgeBand'] = pd.cut(train_data['Age'], 5)

train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# Replace Age with Ordinal (According to AgeBand) and remove AgeBand

for data in combine:    

    data.loc[ data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4

    

train_data = train_data.drop(['AgeBand'], axis=1)

combine = [train_data, test_data]



train_data.head()
# Handle null in Fare

test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

test_data.isnull().sum()
# Cut the Fare (Create FareBand)

train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)

train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# Replace Fare with Ordinal (According to FareBand) and remove FareBand

for data in combine:

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)



train_data = train_data.drop(['FareBand'], axis=1)

combine = [train_data, test_data]

    

train_data.head(10)
# Get Dummies of Age and Fare

train_data = pd.get_dummies(train_data, columns=['Age', 'Fare'])

test_data = pd.get_dummies(test_data, columns=['Age', 'Fare'])

combine = [train_data, test_data]



train_data.head()
train_data.describe()
# Prepare for learning

X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Coefficient of Features

coeff_data = pd.DataFrame(train_data.columns.delete(0))

coeff_data.columns = ['Feature']

coeff_data["Correlation"] = pd.Series(logreg.coef_[0])



coeff_data.sort_values(by='Correlation', ascending=False)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Submission

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)