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
# Matplotlib and seaborn for plotting graphs for EDA

import seaborn as sns

import matplotlib.pyplot as plt
# Load DataSet

titanic=pd.read_csv('../input/train.csv')

# Data Info

print(titanic.head())

print(titanic.columns)

print(titanic.info())

print(titanic.describe())

print(titanic['Sex'].head())
# Count the nulls present in columns

print(titanic.isnull().sum())
# Drop columns not used or convert string into categorical data

sex=pd.get_dummies(titanic['Sex'],drop_first=True)

embark=pd.get_dummies(titanic['Embarked'],drop_first=True)

titanic.drop(['Sex','Embarked','Name','Cabin','Ticket'],axis=1,inplace=True)

train = pd.concat([titanic,sex,embark],axis=1)

train['Age'].fillna(train['Age'].mean(),inplace=True)

print(train.head())
# Almost all female survived whoose Fare>30

sns.scatterplot(x='Fare',y='male',hue='Survived',data=train)

plt.show()
# Pair Plot applicable for low no. of dimensions

sns.pairplot(hue='Survived',data=train,height=2)

plt.show()
# Facetgrid + Distance plot

for idx, feature in enumerate(list(train.columns)[-8:]):

    fg = sns.FacetGrid(train, hue='Survived', height=5)

    fg.map(sns.distplot, feature).add_legend()

    plt.show()
# Check for balanced data

sns.countplot(x='Survived', data=train);

plt.show()
# Count Not Null

train.isnull().count()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(train.drop(['Survived'],axis=1),train['Survived'],test_size=0.30,random_state=23)

print('x_train:',x_train.head())

print('y_train:',y_train.head())
#Create a Regression model

from sklearn.linear_model import LogisticRegression

titanic_model=LogisticRegression()

titanic_model.fit(x_train,y_train)

predict_test=titanic_model.predict(x_test)
# Classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test,predict_test))
# Confusion Matrix

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,predict_test))
# Accuracy=TP+TN/Total=0.8022

# Error_rate=FP+FN/Total=0.1977