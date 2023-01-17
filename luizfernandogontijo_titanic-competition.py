import numpy as np 

import pandas as pd



# plottinf

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')

gender = pd.read_csv('../input/titanic/gender_submission.csv')
train.head()
gender.head()
train.columns
gender.columns
train.dtypes
gender.dtypes
train.isna().sum()
test.isna().sum()
gender.isna().sum()
train.info()

print('_'*40)

test.info()
gender.info()
train.describe(include=['O'])
test.describe(include=['O'])
gender.describe()
test = test.assign(Survived = gender['Survived']) 
columns = ['Name', 'Ticket', 'Cabin']

train.drop(columns, axis=1, inplace=True)
test.drop(columns, axis=1, inplace=True)
train.dropna(subset=['Embarked'], how='all', inplace=True)
train['Age'] = train['Age'].fillna((train['Age'].mean()))
test['Age'] = test['Age'].fillna((train['Age'].mean()))

test['Fare'] = test['Fare'].fillna((train['Fare'].mean()))
PassengerId_train = train['PassengerId']
PassengerId_test = test['PassengerId']
train.drop('PassengerId', axis=1, inplace=True)
test.drop('PassengerId', axis=1, inplace=True)
train_Pclass_Survived = train[['Pclass', 'Survived']].groupby(['Pclass']) #use just Pclass and Survived and group by Pclass
train_Pclass_Survived.mean().sort_values(by='Survived', ascending = False)
train_Age_Survived = train[['Age', 'Survived']].groupby(['Age']) #use just Age and Survived and group by Pclass
train_Age_Survived.mean().sort_values(by='Age', ascending = False)
train_Sex_Survived = train[['Sex', 'Survived']].groupby(['Sex']) #use just Sex and Survived and group by Pclass
train_Sex_Survived.mean().sort_values(by='Sex', ascending = False)
h = sns.FacetGrid(train, col='Survived')

h.map(plt.hist, 'Age')
grid = sns.FacetGrid(train, col='Survived', row='Pclass')

grid.map(plt.hist, 'Age', bins=20)

grid.add_legend()
train.info()
train_oneHot = pd.get_dummies(train)
test_oneHot = pd.get_dummies(test)
test_oneHot.info()
x_train = train_oneHot.drop("Survived", axis=1)

y_train = train_oneHot["Survived"]
x_test = test_oneHot.drop("Survived", axis=1)

y_test = test_oneHot["Survived"]
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

score_log = logreg.score(x_test, y_test)

score_log
svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

score_svc = svc.score(x_test, y_test)

score_svc
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

score_knn = knn.score(x_test, y_test)

score_knn
gNB = GaussianNB()

gNB.fit(x_train, y_train)

y_pred = gNB.predict(x_test)

score_gNB = gNB.score(x_test, y_test)

score_gNB
perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_test)

score_perceptron = perceptron.score(x_train, y_train)

score_perceptron
linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_test)

score_linear_svc = linear_svc.score(x_test, y_test)

score_linear_svc
sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

score_sgd = sgd.score(x_test, y_test)

score_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

score_decision_tree = decision_tree.score(x_test, y_test)

score_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

score_random_forest = random_forest.score(x_test, y_test)

score_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [score_svc, score_knn, score_log, 

              score_random_forest, score_gNB, score_perceptron, 

              score_sgd, score_linear_svc, score_decision_tree]})

models.sort_values(by='Score', ascending=False)
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

score_log = logreg.score(x_test, y_test)

score_log
train = train.assign(PassengerId = PassengerId_train)

test = test.assign(PassengerId = PassengerId_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.info()
submission.to_csv('submission.csv', index = False) 
submission.info()