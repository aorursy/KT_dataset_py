# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualizaition 
#from sklearn.model_selection import train_test_split    # splitting the data into train and test
from sklearn.metrics import roc_curve, auc   # metrics to evaluate models
from sklearn.metrics import accuracy_score   # metrics to evaluate models
from sklearn.linear_model import LogisticRegression  # create logic regression model using  
from sklearn.model_selection import  RandomizedSearchCV    #hyper paramater tunnning


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#load taining & test data
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# Store target variable of training data in a safe place
survived_train = train_df.Survived

# Concatenate training and test sets
titanic = pd.concat([train_df.drop(['Survived'], axis=1), test_df])
titanic.head(5)
# array of columns in dataset
#list(train_df.columns.values)

# list of columns in dataset
list(titanic.columns)
#datatypes in train data
titanic.info()
#statistic table 
titanic.describe()
#shape of dataset(number of columns & rows)
titanic.shape
#missing value in training data
titanic.isnull().sum()
#Filling missing value of cabin with 'x'unknown
titanic['Cabin'].fillna('x', inplace = True)
#filling missing value of age column column with mean
titanic['Age'].fillna(train_df['Age'].mean(), inplace = True)

#filling missing value of age column column with mean
titanic['Fare'].fillna(train_df['Fare'].mean(), inplace = True)
titanic.groupby('Embarked').count()
#filling 2 missing value in 'Embarked' column with 'S' 
titanic['Embarked'].fillna('S', inplace = True)
#distribution of Age 
sns.distplot(titanic['Age'], hist = True) 
#Correlation heatmap
sns.heatmap(titanic.corr(), annot= True)
#creating one hot encoding for categorical feature
list_categ = ['Pclass','Sex','Cabin','Embarked', 'Parch', 'SibSp']
Pclass_dummy = pd.get_dummies(titanic['Pclass'],drop_first=True, prefix='Pclass')
Sex_dummy = pd.get_dummies(titanic['Sex'],drop_first=True, prefix='Sex')
Cabin_dummy = pd.get_dummies(titanic['Cabin'],drop_first=True,prefix='Cabin')
Embarked_dummy = pd.get_dummies(titanic['Embarked'],drop_first=True,prefix = 'Embarked')
Parch_dummy = pd.get_dummies(titanic['Parch'],drop_first=True, prefix='Parch')
SibSp_dummy = pd.get_dummies(titanic['SibSp'],drop_first=True, prefix='SibSp')
# merge all dummified categorical columns
titanic= pd.concat([titanic,Pclass_dummy,Sex_dummy,Cabin_dummy,Embarked_dummy,Parch_dummy,SibSp_dummy],1)
titanic.shape
#delete all non dummified columns
titanic =titanic.drop(['Pclass', 'Sex','Cabin', 'Embarked', 'Parch', 'SibSp'],axis = 1)
titanic.shape
titanic.head()
#prepare train test data
train = titanic.iloc[:891]
test = titanic.iloc[891:].drop(['Name','Ticket'],axis =1)
y = survived_train
X = train.drop(['Ticket','Name'],axis=1)
#create a logistic regression model
logreg = LogisticRegression()
logreg.fit(X,y)
test.shape
X.shape
# rams = { 
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth':np.arange(5,15,1)
# }
# random_search = RandomizedSearchCV(estimator=logreg, param_distributions=params, cv= 5)

# random_search.fit(X, y)
# print (random_search.best_params_)