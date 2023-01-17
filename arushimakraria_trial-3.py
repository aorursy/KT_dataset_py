import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head()

train.shape


train.describe()
train.info()
train.isnull().sum()
test.head()
test.shape
test.info()
test.describe()
test.isnull().sum()
survived = train[train['Survived'] == 1]



not_survived = train[train['Survived'] == 0]



print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))

print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))

print ("Total: " , len(train))
train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
tab = pd.crosstab(train['Pclass'], train['Sex'])

print (tab)



tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Pclass')

plt.ylabel('Percentage')
sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train)
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
train_test_data = [train, test] 



for dataset in train_test_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
train.head()
pd.crosstab(train['Title'], train['Sex'])
for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()
train.Embarked.unique()
train.Embarked.value_counts()
for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head()
for dataset in train_test_data:

    

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train.head()
for dataset in train_test_data:



    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['AgeBand'] = pd.cut(train['Age'], 5)



print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
for dataset in train_test_data:

     dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

     dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train.head()
for dataset in train_test_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['FareBand'] = pd.qcut(train['Fare'], 4)

print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

train.head()
for dataset in train_test_data:

    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1



print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in train_test_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train.head(1)
test.head(1)
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)

train = train.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1)

train.head()
test.head()
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()



X_train.shape, y_train.shape, X_test.shape

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
clf = SVC()

clf.fit(X_train, y_train)

y_pred_svc = clf.predict(X_test)

acc_svc = round(clf.score(X_train, y_train) * 100, 2)

print (acc_svc)
clf = LinearSVC()

clf.fit(X_train, y_train)

y_pred_linear_svc = clf.predict(X_test)

acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)

print (acc_linear_svc)
clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(X_train, y_train)

y_pred_knn = clf.predict(X_test)

acc_knn = round(clf.score(X_train, y_train) * 100, 2)

print (acc_knn)
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred_decision_tree = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

print (acc_decision_tree)
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred_random_forest = clf.predict(X_test)

acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)

print (acc_random_forest)
models = pd.DataFrame({

    'Model': [ 'Support Vector Machines', 'Linear SVC', 

              'KNN', 'Decision Tree', 'Random Forest'],

    

    'Score': [acc_svc, acc_linear_svc, 

              acc_knn,  acc_decision_tree, acc_random_forest ]

    })



models.sort_values(by='Score', ascending=False)

test.head()
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred_random_forest

    })



submission.to_csv('submission.csv', index=False)