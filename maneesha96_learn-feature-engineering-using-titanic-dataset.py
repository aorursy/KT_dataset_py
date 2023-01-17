# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import sklearn
import scipy.stats as stats
import re as re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
train.head()
test.head()
# checking for missing values
full_data = [train,test]
for dataset in full_data:
    print(pd.isnull(dataset).sum()>0)
    break
# checking the impact of the atrributes for the target attribute
# Pclass is categorical attribute
print(train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean())
# Sex is a categorical variable
print(train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean())
# SibSp is a numerical attribute which represent the number of spouse/sibilings
print(train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean())
# Parch is a numerical attribute
print(train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean())
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch']+1
print(train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean())
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean())
# Embarked has missing values
train['Embarked'].value_counts()
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
print(train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CatFare'] = pd.qcut(train['Fare'], 4)
print(train[['CatFare','Survived']].groupby(['CatFare'], as_index=False).mean())
# age is a numerical atrribute. also has missing values. Because of that we need to generate random numbers.
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CatAge'] = pd.cut(train['Age'], 5)

print(train[['CatAge','Survived']].groupby(['CatAge'], as_index=False).mean())

title_arr = []

for i in train['Name']:
    temp_array = i.split(" ")
    if "Mr." in temp_array:
        title_arr.append('Mr') 
    elif "Mrs." in temp_array:
        title_arr.append('Mrs')
    elif "Miss." in temp_array:
        title_arr.append('Miss')
    elif "Master." in temp_array:
        title_arr.append('Master')
    else:
        title_arr.append('Rare')
#print(title_arr)

train['Title'] = title_arr
train['Title'].value_counts()
#print (dataset[['Title','Survived']].groupby(['Title'], as_index=False).mean())
#print(dataset['Title'] )
#dataset.head()
#train_titanic.head()
print (train[['Title','Survived']].groupby(['Title'], as_index=False).mean())
train.head()
title_arr = []

for i in test['Name']:
    temp_arr = i.split(" ")
    if "Mr." in temp_arr:
        title_arr.append('Mr') 
    elif "Mrs." in temp_arr:
        title_arr.append('Mrs')
    elif "Miss." in temp_arr:
        title_arr.append('Miss')
    elif "Master." in temp_arr:
        title_arr.append('Master')
    else:
        title_arr.append('Rare')
#print(title_arr)

test['Title'] = title_arr
test['Title'].value_counts()
#print (dataset[['Title','Survived']].groupby(['Title'], as_index=False).mean())
#print(dataset['Title'] )
#dataset.head()
test.head()

full_data = [train,test]
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].fillna(0)
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1})
    
    title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].map(title_mapping)
 
    dataset['Embarked'] = dataset['Embarked'].fillna(0)
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2})
    
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31.0) & (dataset['Fare'] <= 512.329), 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(float)
    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(float)
    
drop_elements = ['PassengerId','Name','Ticket','Cabin','SibSp','Parch','FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CatAge','CatFare'], axis = 1)
test_passengers = pd.DataFrame(test['PassengerId'])
test = test.drop(drop_elements, axis = 1)

print(train.head())
     
from sklearn.neighbors import KNeighborsClassifier
#import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

train_labels = pd.DataFrame(train['Survived'])
train_features = train.drop(['Survived'], axis = 1)

train_features.head()
X_train, X_test, y_train, y_test = train_test_split(train_features,train_labels,train_size=0.90)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
train_prediction = knn.predict(X_test)
acc = accuracy_score(y_test,train_prediction)
print(acc)
result = knn.predict(test)
print(result)

from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)
result = svc.predict(test)
print(result)
train_prediction = svc.predict(X_test)
acc = accuracy_score(y_test,train_prediction)
print(acc)
submission_file = test
submission_file['PassengerId'] = test_passengers
submission_file['Survived'] = result
drop_el = ['Pclass','Sex','Age','Fare','Embarked','IsAlone','Title']
submission_file = submission_file.drop(drop_el, axis=1)
submission_file = pd.DataFrame(submission_file)
submission_file = submission_file.astype({"PassengerId": str})
submission_file.to_csv("submission_file_svc.csv", index = False)