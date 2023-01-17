# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



# data analysis and wrangling

import numpy as np

import pandas as pd

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Machine Learning Algorithm

from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



# ########

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



# Data Load

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

dataset = [train, test]
# Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.

# Continous: Age, Fare. Discrete: SibSp, Parch.

print("Train Dataset info:")

train.info()

print("\n")

print("Test Dataset info:")

test.info()
train.describe()
train.describe(include=['O'])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
gr = sns.FacetGrid(train, col='Survived')

gr.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', pallete='deep')

grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before:", train.shape, test.shape, dataset[0].shape, dataset[1].shape)



train = train.drop(['Ticket', 'Cabin'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

dataset = [train, test]



print("After:", train.shape, test.shape, dataset[0].shape, dataset[1].shape)
for data in dataset:

    data['Title'] = data['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]

    start_min = 10

    title_names = (data["Title"].value_counts() < start_min)

    data["Title"] = data["Title"].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    

print(pd.crosstab(train['Title'], train['Sex']))

train.head(5)
# Clean Data



for data in dataset:

    data["Age"].fillna(data["Age"].median(), inplace=True)

    data["Fare"].fillna(data["Fare"].median(), inplace=True)

    data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

    

train.isnull().sum()
# Feature Engineering..

for data in dataset:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['IsAlone'] = 1

    data['IsAlone'].loc[data['FamilySize'] > 1] = 0

    

    data['FareBand'] = pd.qcut(data['Fare'], 4)

    data['AgeBand'] = pd.cut(data['Age'].astype(int), 5)

    

train.head()
label = LabelEncoder()

for data in dataset:

    data['Title'] = label.fit_transform(data["Title"])

    data["Fare"] = label.fit_transform(data["FareBand"])

    data["Age"] = label.fit_transform(data["AgeBand"])

    

train.head()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for data in dataset:

    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    data['Embarked'] = data['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)

    

train.head()
for data in dataset:

    data['Age*Class'] = data.Age * data.Pclass

    

train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(5)
train = train.drop(['PassengerId','Name','SibSp','Parch','FamilySize','AgeBand','FareBand'], axis=1)

test = test.drop(['Name','SibSp','Parch','FamilySize','AgeBand','FareBand'], axis=1)

dataset = [train, test]

train.head()
test.head()
# Model, Predict and Solve.

X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression..



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_prediction = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train)* 100, 2)

acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machine



svc = SVC()

svc.fit(X_train, Y_train)

Y_prediction = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# KNN 

knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

Y_prediction = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# GaussianNB Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_prediction = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron.



per = Perceptron()

per.fit(X_train, Y_train)

Y_prediction = per.predict(X_test)

acc_per = round(per.score(X_train, Y_train) * 100, 2)

acc_per
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_prediction = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_prediction = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



tree = DecisionTreeClassifier()

tree.fit(X_train, Y_train)

Y_prediction = tree.predict(X_test)

acc_tree = round(tree.score(X_train, Y_train) * 100, 2)

acc_tree
# Random Forest.



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

acc_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_forest
# Model Evaluation.



models = pd.DataFrame({

    'Model' : ['Support Vector Machine', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Percepton',

              'Stochastic Gradient Decent', 'Linear SVC', 'Decison Tree'],

    'Score' : [acc_svc, acc_knn, acc_log, acc_forest, acc_gaussian, acc_per, acc_sgd, acc_linear_svc, acc_tree]

})



models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_prediction})

submission.to_csv("my_submission.csv", index=False)

print("Your submission saved!")