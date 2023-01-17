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
dataTrain = pd.read_csv("../input/train.csv").drop('Ticket',axis=1)
dataTest = pd.read_csv("../input/test.csv").drop('Ticket', axis=1)
len(dataTrain), len(dataTest)
''' How many NaN Age exist '''
len(dataTrain[dataTrain.Age.isnull()]), len(dataTest[dataTest.Age.isnull()])
''' filling NaN with mean in train and test sets '''
dataTrain.Age.fillna(dataTrain.Age.mean(), inplace = True)
dataTest.Age.fillna(dataTest.Age.mean(), inplace = True)
''' we check how many nan Cabin values exist'''
len(dataTrain[dataTrain.Cabin.isnull()])
''' we delete Cabin feature'''
dataTrain.drop('Cabin', axis = 1, inplace = True)
dataTest.drop('Cabin', axis = 1, inplace = True)
dataTrain.head()
''' we divide the Age feature in quantiles'''
cutted= pd.qcut(dataTrain.Age.values, [0, 0.20, 0.4, 0.6, 0.8, 1.])
pd.value_counts(cutted, sort = False)
dataTrain['Age Quantili'] = cutted
dataTest['Age Quantili'] = pd.cut(dataTest.Age, [0, 20, 28, 29.699, 38, 80 ])

''' we delete also Name feature '''

dataTrain.drop('Name', axis = 1, inplace = True)
dataTest.drop('Name', axis = 1, inplace = True)
dataTrain.head()
dataTrain['Family number'] = dataTrain.SibSp + dataTrain.Parch
dataTest['Family number'] = dataTest.SibSp + dataTest.Parch
dataTest['Fare'].fillna(dataTest['Fare'].mean(), inplace = True)
fare_qbin= pd.qcut(dataTrain.Fare.values,5)

fare_qbin.value_counts()
dataTrain['Fare q_bins'] = fare_qbin
dataTest['Fare q_bins'] = pd.cut(dataTest.Fare, [-0.001, 7.854, 10.5, 21.679, 39.688, 520 ])
dataTrain = pd.get_dummies(dataTrain, columns=['Sex','Embarked'], drop_first = True)
dataTest = pd.get_dummies(dataTest, columns=['Sex','Embarked'], drop_first = True)
dataTest.head()
dataTrain.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataTrain['Fare bins_dummies'] = le.fit_transform(dataTrain['Fare q_bins'])
dataTrain.head()
''' check out the correlation between Pclass and Fare bins_dummies'''

dataTrain['Pclass'].corr(dataTrain['Fare bins_dummies'])
import matplotlib.pyplot as plt
import seaborn as sns
le = LabelEncoder()
dataTest['Fare bins_dummies'] = le.fit_transform(dataTest['Fare q_bins'])
dataTest.head()
dataTrain.head()
le = LabelEncoder()
dataTrain['Age bins_dummies'] = le.fit_transform(dataTrain['Age Quantili'])
le = LabelEncoder()
dataTest['Age bins_dummies'] = le.fit_transform(dataTest['Age Quantili'])
dataTrain.head()
from sklearn.preprocessing import MinMaxScaler
dataTrain_copy = dataTrain.copy()

scaler = MinMaxScaler()
pclass_scaled = scaler.fit_transform(dataTrain_copy.Pclass.values.reshape(-1,1))
dataTrain_copy['Pclass'] = pclass_scaled

scaler2 = MinMaxScaler()
age_dummies_scaled = scaler2.fit_transform(dataTrain_copy['Age bins_dummies'].values.reshape(-1,1))
dataTrain_copy['Age bins_dummies'] = age_dummies_scaled
dataTrain_copy.head(10)
X = dataTrain_copy[['Pclass', 'Family number', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Age bins_dummies']]
X.head(10)
y = dataTrain.Survived

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
''' check out the accuracy for the LogisticRegression'''
logreg = LogisticRegression()
scores = cross_val_score(logreg, X, y, cv = 10, scoring = 'accuracy')

scores.mean()
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,25)
knn_scores = []
for i in k_range:
    knn =  KNeighborsClassifier(n_neighbors = i)
    score_array = cross_val_score(knn, X, y, cv = 10, scoring = 'accuracy')
    knn_scores.append((i,score_array.mean()))
sorted(knn_scores, key = lambda x: x[1], reverse= True)[0]
dataTest.head()
scaler = MinMaxScaler()
pclass_scaled_test = scaler.fit_transform(dataTest.Pclass.values.reshape(-1,1))
dataTest['Pclass'] = pclass_scaled_test
scaler2 = MinMaxScaler()
age_dummies_scaled = scaler2.fit_transform(dataTest['Age bins_dummies'].values.reshape(-1,1))
dataTest['Age bins_dummies'] = age_dummies_scaled
dataTest.head()
X_test = dataTest[['Pclass', 'Family number', 'Sex_male', 'Embarked_Q', 'Embarked_S','Age bins_dummies']]
logreg.fit(X,y)
LG_predictions = logreg.predict(X_test)
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X,y)
knn_predictions = knn.predict(X_test)

titanic_submission = pd.DataFrame(dict(PassengerId = dataTest['PassengerId'].values, LG_predictions = LG_predictions, KNN_predictions = knn_predictions))
titanic_submission.head(10)
''' we choose the KNN model'''
final = titanic_submission.drop(['LG_predictions'], axis = 1)
final.rename(columns = {'KNN_predictions':'Survived'}, inplace = True)
final.head()
final.to_csv('submission.csv', index=False)


