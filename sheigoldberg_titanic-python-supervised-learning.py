import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import re as re



train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

full_data = [train, test]



print (train.info())

print (test.info())
# preview the train data

train.head()
# preview the test data

test.head()
pclass = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

samples = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count()

pclass['Samples'] = samples['Survived']

print (pclass)
sex = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

samples = train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).count()

sex['Samples'] = samples['Survived']

print (sex)
for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

family_size = train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

samples = train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).count()

family_size['Samples'] = samples['Survived']

print (family_size)
for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

is_alone = train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

samples = train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).count()

is_alone['Samples'] = samples['Survived']

print (is_alone)
# the histogram of the data

n, bins, patches = plt.hist(train['Fare'], 10, facecolor='blue', alpha=0.75)

plt.xlabel('Fare')

plt.ylabel('Number of Samples')

plt.title(r'Histogram of Fare for passengers in training data')

plt.show()



for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())



# Create categorical fare values by splitting data in quantiles

train['CategoricalFare'] = pd.qcut(train['Fare'], 10)

categorical_fare = train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

samples = train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).count()

categorical_fare['Samples'] = samples['Survived']

print (categorical_fare)
for dataset in full_data:

    dataset['Age'] = dataset['Age'].fillna(train['Age'].median())

    train['Age'] = train['Age'].fillna(train['Age'].median())

    test['Age'] = test['Age'].fillna(train['Age'].median())

    

# the histogram of the data

n, bins, patches = plt.hist(train['Age'], 10, facecolor='blue', alpha=0.75)

plt.xlabel('Age')

plt.ylabel('Number of Samples')

plt.title(r'Histogram of Age for passengers in training data')

plt.show()



# Create categorical age values by splitting data in quantiles

#train['CategoricalAge'] = pd.qcut(train['Age'], 10, duplicates='drop')

train['CategoricalAge'] = pd.cut(train['Age'], 10)

categorical_age = train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()

samples = train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).count()

categorical_age['Samples'] = samples['Survived']

print (categorical_age)
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.55, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.55) & (dataset['Fare'] <= 7.854), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 8.05), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 10.5), 'Fare'] = 3

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 14.454), 'Fare'] = 4

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 21.679), 'Fare'] = 5

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 27.0), 'Fare'] = 6

    dataset.loc[(dataset['Fare'] > 27.0) & (dataset['Fare'] <= 39.688), 'Fare'] = 7

    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 77.958), 'Fare'] = 8

    dataset.loc[ dataset['Fare'] > 77.958, 'Fare'] = 9

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 8.378, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 8.378) & (dataset['Age'] <= 16.336), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 24.294), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 24.294) & (dataset['Age'] <= 32.252), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 40.21), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 40.21) & (dataset['Age'] <= 48.168), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 56.126), 'Age'] = 6

    dataset.loc[(dataset['Age'] > 56.126) & (dataset['Age'] <= 64.084), 'Age'] = 7

    dataset.loc[(dataset['Age'] > 64.084) & (dataset['Age'] <= 72.042), 'Age'] = 8

    dataset.loc[ dataset['Age'] > 72.042, 'Age'] = 9

    dataset['Age'] = dataset['Age'].astype(int)



# Feature Selection

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize', 'Embarked']

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print (train.head(10))

print (test.head(10))



#train = train.values

#test  = test.values
# import machine learning libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



# define training (X_train) and testing (y_train) sets

X_train = train.drop(['PassengerId','Survived'],axis=1)

y_train = train["Survived"]

X_test = test.drop('PassengerId',axis=1)



print (X_train.head(10))

print (y_train.head(10))
# Logistic Regression

# Submission Results: 0.76555

    

logreg = LogisticRegression()



logreg.fit(X_train, y_train)



#Use the model to make prediction on test data

y_pred = logreg.predict(X_test)

logregsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

logregsubmission.to_csv('logreg.titanic.csv', index=False)



logreg.score(X_train, y_train)
# SVM

# Submission results: 0.76555



svc = SVC()



svc.fit(X_train, y_train)



#Use the model to make prediction on test data

y_pred = svc.predict(X_test)

svcsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

svcsubmission.to_csv('svc.titanic.csv', index=False)



svc.score(X_train, y_train)
# Random Forests

# Submission Results: 0.74162



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, y_train)



#Use the model to make prediction on test data

y_pred = random_forest.predict(X_test)

random_forestsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

random_forestsubmission.to_csv('random_forest.titanic.csv', index=False)



random_forest.score(X_train, y_train)
# K Nearest Neighbors

# Submission results: 0.76076



knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, y_train)



#Use the model to make prediction on test data

y_pred = knn.predict(X_test)

knnsubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

knnsubmission.to_csv('knn.titanic.csv', index=False)



knn.score(X_train, y_train)
# Gaussian Naive Bayes

# Submission results: 0.72727

gaussian = GaussianNB()



gaussian.fit(X_train, y_train)



#Use the model to make prediction on test data

y_pred = gaussian.predict(X_test)

gaussiansubmission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

gaussiansubmission.to_csv('gaussian.titanic.csv', index=False)



gaussian.score(X_train, y_train)