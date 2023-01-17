import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import missingno as missing
%matplotlib inline

sns.set()
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
train.head(10)
train.columns
train = train.reindex_axis(['PassengerId',  'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked','Survived'], axis =1 )
train.head(10)
test.head()
train.info()
train.isnull().sum()
missing.matrix(train)
x_train = train.drop(['PassengerId', 'Cabin','Name','Ticket', 'Survived'], axis=1)
x_train.head()
y_train = train.Survived
y_train.value_counts()
test.head()
x_test= test.drop(['PassengerId', 'Cabin','Name','Ticket'], axis=1)
x_test.head()
PassengerId = test.PassengerId
PassengerId.head()
x_train.isnull().sum()
x_train.Age.fillna(value= x_train.Age.median(), inplace= True)
x_train.Age.isnull().sum()
x_train.Embarked.fillna(value= x_train.Embarked.value_counts().argmax(),inplace= True )
x_train.Embarked.isnull().sum()
missing.matrix(x_train,figsize=(8,6))
missing.matrix(x_test)
x_test.isnull().sum()
x_test.Age.fillna(value= x_train.Age.median(), inplace= True)
x_test.Age.isnull().any()
x_test.Fare.head()
x_test.Fare.fillna(value = x_test.Fare.value_counts().argmax(), inplace=True)
x_test.Fare.isnull().any()
missing.matrix(x_test, figsize=(6,3))
sns.countplot(x_train.Sex)
x_train.head(10)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x_train.Sex =  le.fit_transform(x_train.Sex)
x_train.head()
x_train = pd.get_dummies(x_train, columns=['Pclass', 'Embarked'], drop_first=True)
x_train.head()
x_test = pd.get_dummies(x_test, columns=['Pclass', 'Embarked'], drop_first=True)
x_test.Sex =  le.fit_transform(x_test.Sex)
x_test.head()
from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_test
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)
tree.score(x_train,y_train)
svm = SVC(random_state=0)
svm.fit(x_train,y_train)
svm.score(x_train,y_train)
logistic = LogisticRegression(random_state=0)
logistic.fit(x_train,y_train)
logistic.score(x_train,y_train)
naieve = GaussianNB()
naieve.fit(x_train,y_train)
naieve.score(x_train,y_train)
forest = RandomForestClassifier(n_estimators=100 , random_state=0)
forest.fit(x_train,y_train)
forest.score(x_train,y_train)
params = {'n_estimators':[10, 50,100,150,200,300],'max_depth':[2,3,5,7] , 'verbose':[0,1], 'min_samples_leaf':[1,2,3]}
clf_forst = GridSearchCV(estimator= forest, cv = 5 ,param_grid= params)
clf_forst.fit(x_train,y_train)
clf_forst.best_score_
clf_forst.best_params_
tree_params = {'criterion':['gini','entropy'], 'max_depth':[2,4,5] }
clf_tree = GridSearchCV(estimator= tree, param_grid= tree_params, cv= 5, n_jobs=-1)
clf_tree.fit(x_train,y_train)
clf_tree.best_score_
svm_params = {'C':[1,5,50,100],'degree':[1,3,5,7], 'kernel':['rbf','linear','sigmoid','poly']}
clf_svm = GridSearchCV(estimator=svm, cv=5, param_grid= svm_params)
clf_svm.fit(x_train,y_train)
clf_svm.best_score_
x_train.head()
clf_forst.best_params_
rf = RandomForestClassifier(n_estimators= 200, max_depth=5, random_state=0)
rf.fit(x_train,y_train)
predict = rf.predict(x_test)
predict
submit = pd.DataFrame({'PassengerId':PassengerId, 'Survived':predict})
submit.to_csv('submition.csv', index=False)
submit.head()