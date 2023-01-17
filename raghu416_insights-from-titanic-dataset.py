# Let's import libraries



# for data analysis 

import numpy as np

import pandas as pd



# for visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# to ignore warnings

import warnings

warnings.filterwarnings('ignore')
#import train and test CSV files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Let's have a look at the data

train.head()
train.info()
sex_vs_survived = pd.crosstab(train.Sex, train.Survived)

print(sex_vs_survived)

plt.figure(figsize=(25,10))

sex_vs_survived.plot.bar(stacked=True)
survived_vs_pclass = pd.crosstab([train.Pclass, train.Sex], train.Survived)

print(survived_vs_pclass)

plt.figure(figsize=(25,15))

survived_vs_pclass.plot.bar(stacked = True)
sib = pd.crosstab([train.SibSp, train.Sex], train.Survived)

print(sib)

plt.figure(figsize=(25,10))

sib.plot.bar(stacked = True)
emb = pd.crosstab(train.Embarked, train.Survived)

print(emb)

emb.plot.bar(stacked = True)
emb1 = pd.crosstab([train.Embarked, train.Survived], train.Pclass)

print(emb1)

emb1.plot.bar(stacked = True)
train.head()
name_train = train['Name']

name_train['Title'] = 0

for i in train['Name']:

    name_train['Title']=train['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    

name_test = test['Name']

name_test['Title'] = 0

for i in test['Name']:

    name_test['Title']=test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
print(name_train.Title.unique())

print(name_test.Title.unique())
name_train['Title'] = name_train['Title'].replace(['Miss', 'Ms', 'Mlle', 'Lady'], 'Miss')

name_test['Title'] = name_test['Title'].replace(['Miss', 'Ms', 'Mlle', 'Lady'], 'Miss')

name_test['Title'] = name_test['Title'].replace('Dona', 'Don')
print(name_train.Title.unique())

print(name_test.Title.unique())
train['Title'] = name_train['Title']

test['Title'] = name_test['Title']



title_mean = train.groupby('Title')['Age'].mean()
title_mean
map_title_mean = title_mean.to_dict()

map_title_mean
# fill missing values in the Age column according to title

train.Age = train.Age.fillna(train.Title.map(map_title_mean))

test.Age = test.Age.fillna(train.Title.map(map_title_mean))
print(train.head(15))

print(test.head(15))
train.info()
test.info()
train.drop('Cabin', axis = 1, inplace = True)

train.drop('Name', axis = 1, inplace = True)

train.drop('Ticket', axis = 1, inplace = True)

train.drop('Fare', axis=1, inplace = True)



test.drop('Cabin', axis = 1, inplace = True)

test.drop('Name', axis = 1, inplace = True)

test.drop('Ticket', axis = 1, inplace = True)

test.drop('Fare', axis=1, inplace = True)
print(train.head(10))

print(test.head(10))
title_survival = pd.crosstab(train.Title, train.Survived)

print(title_survival)



plt.figure(figsize=(25,10))

sns.barplot(x='Title', y='Survived', data = train)

plt.xticks(rotation=90);
# peaks for survived/not survived passengers by their age

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age', shade = True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()
sex_mapping = {"male": 0, "female": 1}
embarked_mapping = {'S':0, 'C':1, 'Q':2}
title_mapping = {'Capt': 1,

 'Col': 2,

 'Countess': 3,

 'Don': 4,

 'Dr': 5,

 'Jonkheer': 6,

 'Major': 7,

 'Master': 8,

 'Miss': 9,

 'Mme': 10,

 'Mr': 11,

 'Mrs': 12,

 'Rev': 13,

 'Sir': 14}
train['Sex'] = train['Sex'].map(sex_mapping)

train['Embarked'] = train['Embarked'].map(embarked_mapping)

train['Title'] = train['Title'].map(title_mapping)



test['Sex'] = test['Sex'].map(sex_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)

test['Title'] = test['Title'].map(title_mapping)
print(train.head(10))

print(test.head(10))
train.Embarked = train.Embarked.fillna(0)

test.Embarked = test.Embarked.fillna(0)
test.head()
train.Age = pd.Series(train.Age).astype(int)

train.Embarked = pd.Series(train.Embarked).astype(int)



test.Age = pd.Series(test.Age).astype(int)

test.Embarked = pd.Series(test.Embarked).astype(int)
train.info()

print(train.head(10))

print(test)
from sklearn.model_selection import train_test_split
predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 3000)

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier(n_estimators = 1000)

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission2.csv', index=False)