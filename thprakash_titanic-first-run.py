# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train_df = pd.read_csv("../input/train.csv")

train_df.head()



# Any results you write to the current directory are saved as output.
train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId','Name'], axis=1)

train_df.head()
train_df.describe()
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    

train_df.head()
train_df.Embarked.unique()
train_df.Embarked.value_counts()
train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])

train_df['Embarked'] = train_df['Embarked'].map({'S': 1, 'C' : 2, 'Q' : 3}).astype(int)

train_df.head()
# train_df[pd.isnull(train_df).any(axis=1)]

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].dropna().median())

train_df[pd.isnull(train_df).any(axis=1)]
test_df = pd.read_csv("../input/test.csv")

test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)

test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df['Embarked'] = test_df['Embarked'].fillna(test_df.Embarked.dropna().mode()[0])

test_df['Embarked'] = test_df['Embarked'].map({'S': 1, 'C' : 2, 'Q' : 3}).astype(int)

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].dropna().median())

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



test_df.head()









train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df,col='Survived')

g.map(plt.hist,'Age',bins=20)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
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
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian,  acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)