# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head(5)
data['dept'] = data.sales

data.drop(['sales'], axis = 1, inplace=True)

data.head(5)
salary_dict = {'low' : 35000 , 'medium' : 50000, 'high' : 90000}   

data['estimated_salary'] = data['salary'].map(salary_dict)

#data.head(3)
data.describe()
data['dept'].unique()
correlation = data.corr()

plt.figure(figsize = (8,9))

sns.heatmap(correlation, annot=True, linewidths=.25,cmap="YlGnBu" )

plt.title('correlation between features')
groupby_dept_mean= data.groupby('dept').mean()

groupby_dept_mean
#Visualizing which department gets the most salary

ax = sns.barplot(x="estimated_salary", y="dept", data=data)

#Visualizing which department gets the most salary

from numpy import mean,median

ax = sns.barplot(x="left", y="dept", data=data)
#converting dept column to categorical values

dept_dummies = pd.get_dummies(data['dept'], prefix= 'dept')

data = pd.concat([data,dept_dummies],axis=1)

data.head(5)
data.drop(['dept'],axis=1, inplace=True)



data.drop(['salary'],axis = 1, inplace=True)

data.head(5)
#prediction using Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

y = data.left

X = data.drop(['left'],axis = 1)

clf =  RandomForestRegressor(n_estimators =100,random_state= 101)

clf.fit(X,y)
acc = clf.score(X,y)

print("Decision Tree Regressor accuracy: ",acc)
#k-fold validation 

accuracy = []

n=[100,200,300,400,500,600,700]

for i in n:

    clf = RandomForestRegressor(n_estimators=i, random_state=101)

    clf.fit(X,y)

    accuracy.append(clf.score(X,y))

    

print (accuracy)
from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation

#make a train test split



X_train,X_test,y_train,y_test =  cross_validation.train_test_split(X,y,test_size = 0.25)
print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
clf_dt = DecisionTreeClassifier(max_features= 4,random_state= 101)

clf_dt.fit(X_train,y_train)
accuracy_dt = clf_dt.score(X_test,y_test)

print (" Accuracy of Decision Tree classifier is: ",accuracy_dt)
important_features = clf_dt.feature_importances_

print (important_features)

#data.columns