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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train.csv')
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
train.shape
test.shape
# find missing values from train data

train.isnull().sum()
# description of columns

train.describe().T
sns.barplot(x='Sex',y='Survived',data=train)
# FACET GRID PLOT - FARE vs AGE - Faceting sa per survived column

fg = sns.FacetGrid(train,col='Survived',hue='Survived')
fg.map(sns.scatterplot,'Age','Fare')
sns.jointplot('Age','Fare',data=train)
# survival basis the port of embarkation

fg = sns.FacetGrid(train,col='Embarked',hue='Survived')
fg.map(sns.scatterplot,'Age','Fare')
# pclass vs embarked 

sns.countplot(x='Pclass',data=train,hue='Survived')

# mostly class 3 passengers died 
# class 1 has the lowest death rate compared to other classes
h = sns.FacetGrid(train,col='Pclass',hue='Sex')
h.map(sns.countplot,'Survived')
# fill null values in age to view distribution of age

print(train['Age'].mean())
print(train['Age'].median())

# data is normal. So, we go with filling null values with median
train['Age'] = train['Age'].fillna(28.0)
train['Age'].isnull().sum()
# distribution of AGE variable

sns.distplot(train['Age'])
# impute embared with mode 

train['Embarked'].mode()
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'].isnull().sum()
train['Cabin'] = train['Cabin'].fillna('N')
train['Cabin'].isnull().sum()
plt.figure(figsize=(10,10))
sns.distplot(train['Fare'],color='red')
plt.show()
# combine sibsp and parch to create new column 'family'

train['family'] = train['SibSp'] + train['Parch'] + 1
train.head()
sns.countplot(x="family",hue='Survived',data=train)
# Family column convert to Category 
# 1 - singles
# upto 4 - small
# 5 or more - Large

train['family'].dtypes
def f1(x):
    if (x == 1) :
        return 'single'
    elif ((x <= 4) & (x >=2)) :
        return 'small'
    else :
        return 'Large'
    
train['family_cat'] = list(map(lambda x : f1(x) , train['family'] ))

train.head()
train['family_cat'].value_counts()
a = train['Name'].apply(lambda x : x.split(', ')[-1])
train['titles'] = a.apply(lambda x: x.split('.')[0] )
train.head()
train['titles'].unique()
ignore_titles = list(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',
       'Jonkheer'])
train['titles'] = train['titles'].apply(lambda x : x.replace(x,'others') if x in ignore_titles else x )
train['titles'].unique()
train.head()
# dropping passenger id 

train = train.drop('PassengerId',axis=1)
# drop all unused columns

train = train.drop('Cabin',axis=1)
train = train.drop('Name',axis=1)
train = train.drop('Ticket',axis=1)
train = train.drop('family',axis=1)
train = pd.get_dummies(train , columns = ['Sex','family_cat','titles','Embarked'])
train.head()

from sklearn.model_selection import train_test_split
# dropping survived column 

x = train.drop('Survived',axis=1)
y = train['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25 , random_state=123 )
test = test.drop('Ticket',axis=1)
test = test.drop('Cabin',axis=1)
test = test.drop('PassengerId',axis=1)
test.isnull().sum()
test['Age'].median()
test['Age'] = test['Age'].fillna(27.0)
test[test['Fare'].isnull()]
test[test['Pclass'] == 3].Fare.mean()
test['Fare'] = test['Fare'].fillna(12.459677880184334)
test['family'] = test['SibSp'] + test['Parch'] + 1

test['family_cat'] = list(map(lambda x : f1(x) , test['family'] ))

a = test['Name'].apply(lambda x : x.split(', ')[-1])

test['titles'] = a.apply(lambda x: x.split('.')[0] )

ignore_titles = list(['Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess',
       'Jonkheer'])

test['titles'] = test['titles'].apply(lambda x : x.replace(x,'others') if x in ignore_titles else x )

test.head()
test = pd.get_dummies(test , columns = ['Sex','family_cat','titles','Embarked'])
test.head()
test = test.drop('Name',axis=1)
test = test.drop('family',axis=1)
test = test.drop('titles_Dona',axis=1)
train.head()
test.head()
print(train.shape)
print(test.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
# decision tree

tree = DecisionTreeClassifier()
model_tree = tree.fit(x_train,y_train)
pred_tree = tree.predict(x_test)
accuracy_score(y_test,pred_tree)
# random forest

rf = RandomForestClassifier()
model_rf = rf.fit(x_train,y_train)
pred_rf = model_rf.predict(x_test)
accuracy_score(y_test,pred_rf)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
model_lg = logreg.fit(x_train,y_train)
pred_lg = model_lg.predict(x_test)
accuracy_score(y_test,pred_lg)
#  ADA Boost 

ada = AdaBoostClassifier()
model_ada = ada.fit(x_train,y_train)
pred_ada = model_ada.predict(x_test)
accuracy_score(y_test,pred_ada)
# XGBOOST

xgbc = XGBClassifier()
model_xg = xgbc.fit(x_train,y_train)
pred_xg = model_xg.predict(x_test)
accuracy_score(y_test,pred_xg)
pred_xgtest = model_xg.predict(test)
pred_xgtest
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()
submission['Survived'] = pred_xgtest
sub_df = pd.DataFrame(submission)
sub_df.to_csv('submit.csv',index = False)
