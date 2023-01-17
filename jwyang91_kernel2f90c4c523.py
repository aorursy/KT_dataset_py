train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train.head()
print(train.shape)
print(train.info())
print(test.shape)
print(test.info())
full_set = [train,test]
for data in full_set:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.')
#Title Extraction

train.head(5)
pd.crosstab(train['Title'], train['Sex'])
#직위/성별 관련 데이터 정제

for data in full_set:
    data['Title'] = data['Title'].replace(['Capt', 'Col',  'Don', 'Dr', 'Jonkheer','Major', 'Rev', 'Sir'], 'Mr') #융커, 경 등 남성 타이틀 Mr. 단, Dr중 여성이 한명 있음.
    data['Title'] = data['Title'].replace(['Mlle'], 'Miss') #마드모아젤 -> Miss
    data['Title'] = data['Title'].replace(['Mme'], 'Mrs') #마담 -> Mrs
    data['Title'] = data['Title'].replace(['Ms'], 'Miss') #Ms -> Miss 로 통일
    data['Title'] = data['Title'].replace(['Dona','Countess','Lady'], 'Mrs') #백작부인 등 귀부인 타이틀 Mrs

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# 성별데이터 정리
for data in full_set:
    data['Sex'] = data['Sex'].astype(str)
#Age Binning
for data in full_set:
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4
    data['Age'] = data['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])
for data in full_set:
    data['Fare']=data['Fare'].fillna(13.68)
for data in full_set:
    data.loc[ data['Fare'] <= 7.85, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.85) & (data['Fare'] <= 10.5), 'Fare'] = 1
    data.loc[(data['Fare'] > 10.5) & (data['Fare'] <= 21.68), 'Fare'] = 2
    data.loc[(data['Fare'] > 21.68) & (data['Fare'] <= 39.69), 'Fare'] = 3
    data.loc[ data['Fare'] > 39.70, 'Fare'] = 4
    data['Fare'] = data['Fare'].astype(int)
for data in full_set:
    data['Family'] = data['Parch'] + data['SibSp']
    data['Family'] = data['Family'].astype(int)    
features_drop = ['Name','Ticket','Cabin','SibSp','Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)

train.head(5)

train = train.drop(['PassengerId'], axis=1)

train.head(5)
test.head(5)
# One-hot-encoding for categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.utils import shuffle
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
def train_n_test(model):
    model.fit(train_data,train_label)
    prediction = model.predict(test_data)
    accuracy = round (model.score(train_data,train_label)*100,2)
    print("Accuracy : ", accuracy, "%")
    return prediction
# Logistic Regression
log_pred = train_n_test(LogisticRegression())
# SVM
svm_pred = train_n_test(SVC())
#kNN
knn_pred_4 = train_n_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_n_test(RandomForestClassifier(n_estimators=100))
# Navie Bayes
nb_pred = train_n_test(GaussianNB())
submission = pd.DataFrame({
"PassengerId": test["PassengerId"],
"Survived": rf_pred
})

submission.to_csv('submission_rf.csv', index=False)