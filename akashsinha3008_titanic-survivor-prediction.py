# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.corr()
train.info()
train.describe()
def nulltable(training,testing):
    print(pd.isnull(training).sum())
    print(" ")
    print(pd.isnull(testing).sum())
    
nulltable(train,test)
train = train.drop(['Cabin'],axis = 1)
test = test.drop(['Cabin'],axis = 1)
nulltable(train,test)
#Replacing missing age with mean age
train.Age.fillna(train.Age.mean(), inplace=True)
test.Age.fillna(test.Age.mean(), inplace=True)
nulltable(train,test)
#Replacing missing Fare with mean Fare
train.Fare.fillna(train.Fare.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)

nulltable(train,test)
train = train.drop(['PassengerId','Name','Ticket'],axis = 1)
PassengerId = test['PassengerId']
test = test.drop(['PassengerId','Name','Ticket'],axis = 1)
train.head()
test.head()
#Creating bins for Age

def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4:
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'young_adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior citizen'
    else:
        a = 'old'
    return a
train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = train['Age'].map(age_group_fun)
#Creating family size feature
train['family_size'] = train.SibSp + train.Parch + 1
test['family_size'] = train.SibSp + train.Parch + 1
def family_size_group(size):
    a = ''
    if size <= 1:
        a = 'loner'
    elif size <= 4:
        a = 'small'
    else:
        a = 'large'
    return a
train['family_group'] = train['family_size'].map(family_size_group)
test['family_group'] = test['family_size'].map(family_size_group)
#Calculated Fare feature
train['calculated_fare'] = train.Fare /train.family_size
test['calculated_fare'] = test.Fare/test.family_size
#Fare group
def fare_group(fare):
    a =''
    if fare <= 4:
        a = 'very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = 'very_high'
    return a
train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)
#dummy for train data
train = pd.get_dummies(train,columns=['Pclass','Sex','Embarked','age_group','family_group','fare_group'],drop_first=True)
train.drop(['Age','SibSp','Parch','family_size','calculated_fare','Fare'],axis = 1,inplace = True)
train.head()
#dummy for test data
test = pd.get_dummies(test,columns=['Pclass','Sex','Embarked','age_group','family_group','fare_group'],drop_first=True)
test.drop(['Age','SibSp','Parch','family_size','calculated_fare','Fare'],axis = 1,inplace = True)
test.head()
#Separating independent and dependent variables
X = train.drop(['Survived'], axis=1)
y = train["Survived"]
#Splitting training dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = .2, random_state = 0)
x_train.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
test = sc.transform(test)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = logreg, X = x_train, y = y_train, cv = 10, n_jobs = -1)
logreg_accy = accuracies.mean()
print (round((logreg_accy),3))
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

pred_tree = tree.predict(x_test)

tree.score(x_train, y_train)
tree_score = round(tree.score(x_train, y_train),3)
tree_score
y_final_pred = tree.predict(test)
output = pd.DataFrame(PassengerId)
output['Survived'] = y_final_pred
output.to_csv('Gender_submission.csv',index=False)