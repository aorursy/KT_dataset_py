import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer
train_df = pd.read_csv('../input/train.csv') # training portion

test_df = pd.read_csv('../input/test.csv')

combine_df = [train_df, test_df] # all data 

train_df.head()
print(train_df.columns.values) #get the list of all column names
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include='O')
fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(train_df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
print("Before", train_df.shape, test_df.shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for obs in combine:

    obs['Title'] = obs.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for obs in combine:

    obs['Title'] = obs['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    obs['Title'] = obs['Title'].replace('Mlle', 'Miss')

    obs['Title'] = obs['Title'].replace('Ms', 'Miss')

    obs['Title'] = obs['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Rare": 2, "Master": 3, "Miss": 4, "Mrs": 5}

for obs in combine:

    obs['Title'] = obs['Title'].map(title_mapping)

    obs['Title'] = obs['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for obs in combine:

    obs['Sex'] = obs['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
my_imputer = SimpleImputer()

train_df['Age'] = pd.DataFrame(my_imputer.fit_transform(train_df['Age'].values.reshape(-1,1)))

test_df['Age'] = pd.DataFrame(my_imputer.fit_transform(test_df['Age'].values.reshape(-1,1)))
my_imputer_cat = SimpleImputer(strategy = 'most_frequent')

train_df['Embarked'] = pd.DataFrame(my_imputer_cat.fit_transform(train_df['Embarked'].values.reshape(-1,1)))

test_df['Embarked'] = pd.DataFrame(my_imputer_cat.fit_transform(test_df['Embarked'].values.reshape(-1,1)))
test_df['Fare'] = pd.DataFrame(my_imputer.fit_transform(test_df['Fare'].values.reshape(-1,1)))
combine = [train_df, test_df]
pd.crosstab(train_df['Survived'], train_df['Embarked'])
for obs in combine:

    obs['Embarked'] = obs['Embarked'].map( {'S': 0, 'Q': 1, 'C': 2} ).astype(int)
train_df.head()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines

from sklearn.svm import SVC, LinearSVC



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
from sklearn.svm import  LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
from sklearn.ensemble import GradientBoostingClassifier

sgd = GradientBoostingClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
output = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

output.to_csv('submission.csv', index=False)