# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style

import matplotlib as mpl

import matplotlib.pylab as pylab





# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics





#misc libraries

import random

import time



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)

from subprocess import check_output
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.shape
train.describe()
test.head()
test.shape
test.describe()
#Checking missing values 

total = train.isnull().sum().sort_values(ascending=False)

percent_1 = train.isnull().sum()/train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
#Checking missing values 

total = test.isnull().sum().sort_values(ascending=False)

percent_1 = test.isnull().sum()/test.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
data = [train, test]



for dataset in data:    

    #complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



    #complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



    #complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
#delete the cabin feature/column and others previously stated to exclude in train dataset    

train = train.drop(['PassengerId','Cabin', 'Ticket'], axis=1)

test = test.drop(['Cabin', 'Ticket'], axis=1)
#Checking missing values again

total = train.isnull().sum().sort_values(ascending=False)

percent_1 = train.isnull().sum()/train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
#Checking missing values again

total = test.isnull().sum().sort_values(ascending=False)

percent_1 = test.isnull().sum()/test.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
data = [train, test]



for dataset in data:    

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    # replace titles with a more common title as Other

    dataset['Title'] = dataset['Title'].replace(['Dr','Rev', 'Col', 'Major', 'Mlle', 'Sir', 'Ms', 'Don', 'Lady', 'Mme' , 'the Countess' ,'Jonkheer' , 'Capt', 'Dona'], 'Other')
train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)
print(train['FamilySize'].value_counts())

print(train['IsAlone'].value_counts())

print(train['Title'].value_counts())
print(test['FamilySize'].value_counts())

print(test['IsAlone'].value_counts())

print(test['Title'].value_counts())
print(train.Survived.unique())
print(train.Pclass.unique())

print(test.Pclass.unique())
print(train.Sex.unique())

print(test.Sex.unique())
print(train.Age.unique())

print(test.Age.unique())
print(train.SibSp.unique())

print(test.SibSp.unique())
print(train.Parch.unique())

print(test.Parch.unique())
print(train.Fare.unique())

print(test.Fare.unique())
print(train.Embarked.unique())

print(test.Embarked.unique())
print(train.FamilySize.unique())

print(test.FamilySize.unique())
print(train.IsAlone.unique())

print(test.IsAlone.unique())
print(train.Title.unique())

print(train.Title.unique())
#preview data again

train.head()
#preview data again

test.head()
plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age')

plt.ylabel('# of Passengers')

plt.legend()
plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()
fig, saxis = plt.subplots(3, 3,figsize=(17,12))



sns.countplot(x='Survived', data=train, ax = saxis [0,0])

sns.barplot(x='Pclass', y='Survived', data=train, ax = saxis [0,1])

sns.countplot(x='Sex', hue='Survived', data=train, ax = saxis [0,2])



sns.barplot(x='SibSp', y='Survived', data=train, ax = saxis [1,0])

sns.barplot(x='Parch', y='Survived', data=train, ax = saxis [1,1])

sns.barplot(x='Embarked', y='Survived', data=train, ax = saxis [1,2])



sns.barplot(x='FamilySize', y='Survived', data=train, ax = saxis [2,0])

sns.barplot(x='IsAlone', y='Survived', data=train, ax = saxis [2,1])

sns.barplot(x='Title', y='Survived', data=train, ax = saxis [2,2])
fig, saxis = plt.subplots(2,3,figsize=(17,15))



sns.countplot(x='Pclass',hue='Sex',data=train, ax = saxis [0,0])

sns.countplot(x='Pclass',hue='SibSp',data=train, ax = saxis [0,1])

sns.countplot(x='Pclass',hue='Parch',data=train, ax = saxis [0,2])



sns.countplot(x='Pclass',hue='Embarked',data=train, ax = saxis [1,0])

sns.countplot(x='Pclass',hue='FamilySize',data=train, ax = saxis [1,1])

sns.countplot(x='Pclass',hue='IsAlone',data=train, ax = saxis [1,2])

fig, saxis = plt.subplots(2,3,figsize=(17,15))



sns.countplot(x='Sex',hue='Embarked',data=train, ax = saxis [0,0])

sns.countplot(x='Sex',hue='FamilySize',data=train, ax = saxis [0,1])

sns.countplot(x='Sex',hue='IsAlone',data=train, ax = saxis [0,2])





sns.countplot(x='SibSp',hue='Embarked',data=train, ax = saxis [1,0])

sns.countplot(x='Parch',hue='Embarked',data=train, ax = saxis [1,1])

sns.countplot(x='Embarked',hue='FamilySize',data=train, ax = saxis [1,2])
cor = train.corr()

plt.figure(figsize=(12,6))

sns.heatmap(cor, annot=True)

plt.title('Correlation')

plt.show()
train.info()
test.info()
genders = {"male": 0, "female": 1}

data = [train, test]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
ports = {"S": 1, "C": 2, "Q": 3}

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
title = {'Mr':1, 'Mrs':2 ,'Miss':3, 'Master':4, 'Other':5} 

data = [train, test]



for dataset in data:

    dataset['Title'] = dataset['Title'].map(title)
data = [train, test]



for dataset in data:

    mean = train["Age"].mean()

    std = test["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train["Age"].astype(int)

train["Age"].isnull().sum()
data = [train, test]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
#preview data again

train.head()
#preview data again

test.head()
print(train.Embarked.unique())

print(test.Embarked.unique())
print(train.Sex.unique())

print(test.Sex.unique())
print(train.Fare.unique())

print(test.Fare.unique())
print(train.Age.unique())

print(test.Age.unique())
print(train.Title.unique())

print(test.Title.unique())
print('Train columns with null values: \n', train.isnull().sum())

print("-"*10)

print (train.info())

print("-"*10)



print('Test/Validation columns with null values: \n', test.isnull().sum())

print("-"*10)

print (test.info())

print("-"*10)



train.describe(include = 'all')
from sklearn import preprocessing, model_selection, metrics, feature_selection

X_train = train.loc[:, train.columns != 'Survived']

Y_train = train.loc[:, train.columns == 'Survived']

X_test  = test.drop("PassengerId", axis=1).copy()

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

Y_pred = dt.predict(X_test)

acc_dt = round(dt.score(X_train, Y_train) * 100, 2)
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
gbm = xgb.XGBClassifier()

gbm.fit(X_train, Y_train)

Y_pred = gbm.predict(X_test)

acc_gbm = round(gbm.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({

    'Model': ['Stochastic Gradient Descent','Random Forest', 'Logistic Regression', 'Decision Tree', 'KNN', 'Gaussian Naive Bayes', 'Perceptron', 'Linear Support Vector Machine', 'XGBoost' ],

    'Score': [acc_sgd, acc_random_forest, acc_log, acc_dt, acc_knn, acc_gaussian, acc_perceptron,acc_linear_svc, acc_gbm]})



result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score'), 

result_df
#prepare data for modeling

print(test.info())

print("-"*10)

#data_val.sample(10)
subm = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

subm.to_csv("subm.csv", index=False)
subm