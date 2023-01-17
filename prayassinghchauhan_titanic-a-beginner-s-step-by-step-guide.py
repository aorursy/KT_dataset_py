import sys

import pandas as pd

import numpy as np

import matplotlib

import sklearn         # collection of machine learning algorithms
import IPython         # pretty printing of dataframes in Jupyter notebook

from IPython import display



import scipy as sp     # collection of functions for scientific computing and advance mathematics
# misc libraries 

import random

import time



# ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)
from subprocess import check_output

#print(check_output(["ls", "E:/SOIL/Data Science/Kaggle/Titanic/titanic"]).decode("utf8"))
# importing the data 



data_raw = pd.read_csv('../input/titanic/train.csv')

data_val = pd.read_csv('../input/titanic/test.csv')
# Checking null values

print('Train columns with null values:\n',data_raw.isnull().sum())

print('-'*40)

print('Test columns with null values:\n',data_val.isnull().sum())
#COMPLETING

# replacing the null values from median(for quantitative) and modes(for qualitative)



data_cleaner = [data_raw,data_val]

for data in data_cleaner:

    median_age = data['Age'].median()

    data['Age'].fillna(median_age,inplace = True)

    

    mode_emb = data['Embarked'].mode()[0]

    data['Embarked'].fillna(mode_emb,inplace = True)

    

    median_fare = data['Fare'].median()

    data['Fare'].fillna(median_fare,inplace = True)
print(data_raw.isnull().sum())

print('-'*40)

print(data_val.isnull().sum())
print(mode_emb)

print(median_age)

print(median_fare)
# DELETING

# deleting the unwanted columns/features from train dataset

# dropping passengerID(because IDs are of no use in analysis)

# dropping ticket(because ticket is in mixed type qualitative plus quantitave, hence can't be analysed)

# droping cabin(because of too many blank/null data)



data_raw.drop(['PassengerId','Ticket','Cabin'],axis=1,inplace = True) #here we dropped the unwanted variables/columns/features and previewed the first 5 columns using .head() function
# CREATING

# creating the new variables as per our understanding of the data

# we can create family size and we can check weather the person is alone or not



for data in data_cleaner:

    data['FamSize'] = data['SibSp']+data['Parch']+1
for data in data_cleaner:

    data['IsAlone'] = 1 # if alone initialize 1/yes

    data['IsAlone'].loc[data['FamSize']>1] = 0 # if family size is '>1', then not alone so, update 0/no
data_raw.head()
# splitting the names to see the name titles



for data in data_cleaner:

    data['Title'] =  data['Name'].str.split(',',expand = True)[1] #while name splitted by ','

     #keeping only the value at 1st index, that's why [1]
data_raw['Title'].head()
for data in data_cleaner:

    data['Title'] = data['Name'].str.split(',',expand = True)[1].str.split('.',expand=True)[0] #keeping only the value at 1st index, that's why [0]
data_raw['Title'].value_counts()
data_raw.head()
#for data in data_cleaner:

#    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

# 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

#

#    data['Title'] = data['Title'].replace('Mlle', 'Miss')

#    data['Title'] = data['Title'].replace('Ms', 'Miss')

#    data['Title'] = data['Title'].replace('Mme', 'Mrs')
title_names = (data_raw['Title'].value_counts()<10)

data_raw['Title'] = data_raw['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
data_raw['Title'].value_counts()
data_raw.head()
for data in data_cleaner:

    data['FareBin'] = pd.cut(data['Fare'].astype(int),bins=4)

    data['AgeBin'] = pd.cut(data['Age'].astype(int),bins=5)
data_raw.head()
# CONVERT



from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

for data in data_cleaner:

    data['SexCode'] = label.fit_transform(data['Sex'])

    data['EmbarkedCode'] = label.fit_transform(data['Embarked'])

    data['TitleCode'] = label.fit_transform(data['Title'])

    data['FareBinCode'] = label.fit_transform(data['FareBin'])

    data['AgeBinCode'] = label.fit_transform(data['AgeBin'])
data_raw.head()
data_raw.drop(['Name','FareBin','AgeBin'],axis=1,inplace=True)
data_raw.head()
data_raw.drop(['SibSp','Parch'],axis=1,inplace=True)
data_raw.head()
data_raw.drop(['Sex','Title'],axis=1,inplace=True)

data_raw.head()
data_raw.drop(['FareBinCode','AgeBinCode','Embarked'],axis=1,inplace=True)

data_raw.head()
data_val.head()
data_val.drop(['Name','Sex','SibSp','Parch','Ticket','Cabin','Title','FareBin','AgeBin','FareBinCode','AgeBinCode','Embarked'],axis=1,inplace=True)
X_train = data_raw.drop('Survived',axis=1)

Y_train = data_raw['Survived']

X_test = data_val.drop('PassengerId',axis=1).copy()

X_train.shape,Y_train.shape,X_test.shape
print(X_train)

print('-'*40)

print(Y_train)

print('-'*40)

print(X_test)

print('-'*40)
# Logistic Regression



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=30000)

logreg.fit(X_train,Y_train)
predictions = logreg.predict(X_test)
accuracy = round(logreg.score(X_train,Y_train)*100,3)

print(accuracy)
# Support Vector Machines



from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
# Gradient Boosting



from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(X_train, Y_train)

pred = gbk.predict(X_test)

acc_gbk = round(gbk.score(X_train, Y_train) * 100, 2)

print(acc_gbk)
# Random Forest



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 3)

acc_random_forest
# Decision Tree



from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 3)

acc_decision_tree
# KNN



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression','Random Forest', 'Naive Bayes',

              'Decision Tree','Gradient Boosting'],

    'Score': [acc_svc, acc_knn, accuracy,acc_random_forest, acc_gaussian,acc_decision_tree,acc_gbk]})

models.sort_values(by='Score', ascending=False)
ids = data_val['PassengerId']

predictions = random_forest.predict(data_val.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)