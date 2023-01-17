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
# Load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

combine = [train_data, test_data]



print(train_data.columns.values)
# Distribution of numeric data

train_data.describe()



#train_data.describe(percentile = [.62, .63])
# Distribution of String data

train_data.describe(include = [np.object])
# Correlation between PClass and Survived

train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Correlation between Sex and Survived

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Visualization analysis for correlation between Age and Survived

g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)



#sns.countplot(x="Age", hue="Survived", data=train_data)
# Further visualization analysis for correlation between Age, Pclass and Survived

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
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



#train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Group the Title

for data in combine:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Change Title to Ordinal

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data in combine:

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# Drop Name and ID

train_data = train_data.drop(['Name', 'PassengerId'], axis=1)

test_data = test_data.drop(['Name'], axis=1)

combine = [train_data, test_data]

train_data.shape, test_data.shape
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

    data.loc[ data['Age'] > 64, 'Age']

    

train_data = train_data.drop(['AgeBand'], axis=1)

combine = [train_data, test_data]



train_data.head()
# Combine SibSp and Parch

for data in combine:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Create isAlone

for data in combine:

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
# Drop Parch, SibSp and FamilySize

train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_data, test_data]



train_data.head()
# Artificial Feature for Age * Class (how important a Person)

for data in combine:

    data['Age*Class'] = data.Age * data.Pclass



train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
# Handle null in Embark

freq_port = train_data.Embarked.dropna().mode()[0]

for data in combine:

    data['Embarked'] = data['Embarked'].fillna(freq_port)

    

train_data.isnull().sum()
# Change Embarked to Ordinal

for data in combine:

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



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
test_data.head(10)
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
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Model Evaluation

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
# Submission

submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)