# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
print(data_train.info())
print('-'*15)
print(data_test.info())

data = [data_train, data_test]
# data preprocessing
# 1. completeing dataset
from sklearn.preprocessing import Imputer
imputer = Imputer("NaN", "mean")
for d in data:
    d['Age'] = imputer.fit_transform(d['Age'].values.reshape(-1,1))
    d['Fare'] = imputer.fit_transform(d['Fare'].values.reshape(-1,1))
    d['Embarked'] = d['Embarked'].fillna('S')
# print(data_train.info(), '\n', data_test.info())
    
# 2. creating new features familySize, isAlone
import re
def titleSearch(name):
    title = re.search('([A-Za-z]+\.)', name)
    if title:
        return title.group(0)
    return ""

for d in data:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
    d['Title'] = d['Name'].apply(titleSearch)
    d['Title'] = d['Title'].replace(['Capt.', 'Col.', 'Countess.', 'Don.', 'Jonkheer.', 'Rev.'], 'Rare.')
    d['Title'] = d['Title'].replace(['Mlle.', 'Mme.', 'Ms.', 'Lady.'], 'Miss.')
    d['Title'] = d['Title'].replace(['Sir.', 'Master.'], 'Major.')

# print(data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
# print(pd.crosstab(data_train['Title'], data_train['Sex']))

for d in data:
    d.loc[(d['FamilySize'] == 1), 'isAlone'] = 1
    d.loc[d['FamilySize'] > 1, 'isAlone'] = 0
# print(data_train[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean())

# 3. Creating dummy variable dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for d in data:
    d['Sex'] = le.fit_transform(d['Sex'])
    d['Embarked'] = le.fit_transform(d['Embarked'])
    d['Title'] = le.fit_transform(d['Title'])
    d['CatFare'] = pd.qcut(data_train['Fare'], 4)
    d['CatAge'] = pd.cut(data_train['Age'], 5)
    pd.cut(data_train['Age'], 4)
print(data_train.head(10))

# 4. categorical variables fare, Age
print(data_train[['CatFare', 'Survived']].groupby(['CatFare'], as_index=False).mean())
print(data_train[['CatAge', 'Survived']].groupby(['CatAge'], as_index=False).mean())

for d in data:
    d.loc[(d['Fare'] > 0) & (d['Fare'] <= 7), 'Fare'] = 0
    d.loc[(d['Fare'] > 7) & (d['Fare'] <= 14), 'Fare'] = 1
    d.loc[(d['Fare'] > 14) & (d['Fare'] <= 31), 'Fare'] = 2
    d.loc[(d['Fare'] > 31), 'Fare'] = 3
    
    d.loc[(d['Age'] > 0) & (d['Age'] <= 16), 'Age'] = 0
    d.loc[(d['Age'] > 16) & (d['Age'] <= 32), 'Age'] = 1
    d.loc[(d['Age'] > 32) & (d['Age'] <= 48), 'Age'] = 2
    d.loc[(d['Age'] > 48) & (d['Age'] <= 64), 'Age'] = 3
    d.loc[(d['Age'] > 64), 'Age'] = 4
    

y = data_train['Survived']
print(data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

drop_col = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'CatFare', 'CatAge', 'Parch', 'SibSp']

data_train = data_train.drop(drop_col+['Survived'], 1)
data_test = data_test.drop(drop_col, 1)
# print(data_train)

# get_dummies of title
# for d in data:
data_train = pd.get_dummies(data_train, columns=['Title'], prefix='Title', drop_first=True)
data_test =  pd.get_dummies(data_test, columns=['Title'], prefix='Title', drop_first=True)
#     print(d.head(10))

# data[0] = data[0].append(cat_title)
print(data_train.head(10))
# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder(categorical_features=[4, 6])
# data_train = onehotencoder.fit_transform(data_train)
# print(data_train)
# split dataset in training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_train.values, y.values, test_size=0.20, random_state=42)
# print(X_train, y_train)

# classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

classifiers = [LogisticRegression(random_state=0), 
              KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', p=2),
              SVC(kernel='poly', random_state=0),
              SVC(kernel='rbf', random_state=0),
              GaussianNB(),
              DecisionTreeClassifier(criterion='entropy', random_state=0),
              RandomForestClassifier(criterion="entropy", random_state=0)]

acc_dict = {}
for clf in classifiers:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    name = clf.__class__.__name__
#     print(score, name)
    if name in acc_dict:
        acc_dict[name] = score
    else:
        acc_dict[name] = score

print(acc_dict)
# predict data using KNN
classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', p=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(data_test.iloc[:, 0:-1].values)
