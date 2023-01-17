import pandas as pd

import numpy as np

import random as rnd





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train, test]
train.describe()
train.info()
train.describe(include=['O'])
train.head()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
def child(passenger):

    age,sex = passenger

    if age<16:

        return 'child'

    else:

        return sex

train['person'] = train[['Age', 'Sex']].apply(child,axis=1)

test['person'] = test[['Age','Sex']].apply(child,axis=1)
sns.countplot(train['Pclass'],hue=train['person'])
persons = train['person'].value_counts()

persons
p = test['person'].value_counts()

p
persons.plot.bar()
age_survived = sns.FacetGrid(train,col='Survived')

age_survived.map(plt.hist,'Age', bins=20)
train["Survived"] = train.Survived.map({0: "no", 1: "yes"})

sns.countplot('Survived',data=train)
train['Survived'].value_counts(normalize=True) * 100
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, col='Pclass', hue='Survived', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train['Age'] = train['Age'].replace('NaN',train.Age.mean())

test['Age'] = test['Age'].replace('NaN',test.Age.mean())
train["Age"] = train['Age'].astype(int)

test["Age"] = test['Age'].astype(int)
train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]
train.head()

test.head()
train['person'] = train['person'].map( {'child':0,'female': 1, 'male': 2} ).astype(int)

test['person'] = test['person'].map( {'child':0,'female': 1, 'male': 2} ).astype(int)

train.head()
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



train['Title'] = train['Title'].replace('Mlle', 'Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



test['Title'] = test['Title'].replace('Mlle', 'Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

train['Title'] = train['Title'].map(title_mapping)

train['Title'] = train['Title'].fillna(0)



train.head()
title_mappings = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

test['Title'] = test['Title'].map(title_mappings)

test['Title'] = test['Title'].fillna(0)



test.head()
train[['Title', 'Survived']].groupby(train['Title'], as_index=False).mean()
train.head()
train = train.drop(['Name', 'PassengerId'], axis=1)

test = test.drop(['Name'], axis=1)

combine = [train, test]
train.shape, test.shape
train = train.drop(['Sex'],axis=1)
train.head()
train['Survived'] = train['Survived'].map( {'no':0,'yes': 1} ).astype(int)
train.head()
test = test.drop(['Sex'], axis=1)
test.head()
train["Alone"] = train.Parch + train.SibSp

train["Alone"].head()
test["Alone"] = test.Parch + test.SibSp

test["Alone"].head()
train["Alone"].loc[train["Alone"] != 0] = 'With Family'

train["Alone"].loc[train["Alone"] == 0] = 'Alone'
test["Alone"].loc[test["Alone"] != 0] = 'With Family'

test["Alone"].loc[test["Alone"] == 0] = 'Alone'
freq_port = train.Embarked.dropna().mode()[0]

freq_port
train['Embarked'] = train['Embarked'].fillna(freq_port)

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()
train['Alone'] = train['Alone'].map( {"Alone":0,'With Family':1}).astype(int)
test['Alone'] = test['Alone'].map( {"Alone":0,'With Family':1}).astype(int)
train.head()
test.head()
# train['AgeBand'] = pd.cut(train['Age'], 5)

# train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# test.loc[ test['Age'] <= 16, 'Age'] = 0

# test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1

# test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2

# test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3

# test.loc[ test['Age'] > 64, 'Age']
test.head()
test = test.drop(['Parch', "SibSp"],axis=1)
train = train.drop(['Parch', "SibSp"],axis=1)
train.shape,test.shape
train["Fare"] = train.Fare.astype(int)
train.head()
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0

train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1

train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2

train.loc[ train['Fare'] > 31, 'Fare'] = 3

train['Fare'] = train['Fare'].astype(int)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.loc[ test['Fare'] <= 7.91, 'Fare'] = 0

test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1

test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2

test.loc[ test['Fare'] > 31, 'Fare'] = 3

test['Fare'] = test['Fare'].astype(int)
test.head()
train.head()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape


logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
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
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
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
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
submission.to_csv('/home/ubuntu/Desktop/titanic.csv', index=False)