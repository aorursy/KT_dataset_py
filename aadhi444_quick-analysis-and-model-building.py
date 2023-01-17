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
# Getting all the required packages
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
# Reading the train and test files
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
print('Shape of train data: ', train_data.shape)
print('Shape of test data: ', test_data.shape)
# getting a look at what the train data looks like

train_data.head()
# Getting the overall description of the data that we have
train_data.info()
print('\n----------------------------------\n')
test_data.info()
train_data = train_data.drop(['Name', 'Ticket', 'PassengerId','Cabin'], axis =1)
test_data = test_data.drop(['Name', 'Ticket','Cabin'], axis = 1)
# Now its time for imputation
# Mode for categorical variable and mean for continuous variable
age_mean = train_data['Age'].mean()
train_data['Age'] = train_data['Age'].fillna(age_mean)
test_data['Age'] = test_data['Age'].fillna(age_mean)

fare_mean = train_data['Fare'].mean()
test_data['Fare'] = test_data['Fare'].fillna(fare_mean)

embarked_mode = train_data['Embarked'].mode()
train_data['Embarked'] = train_data['Embarked'].fillna(embarked_mode)
test_data['Embarked'] = test_data['Embarked'].fillna(embarked_mode)
# Let's see what's there in the PClass
sns.distplot(train_data['Pclass'])
train_data.Pclass[train_data.Pclass == 1] = 'First'

train_data.Pclass[train_data.Pclass == 2] = 'Second'

train_data.Pclass[train_data.Pclass == 3] = 'Third'

test_data.Pclass[test_data.Pclass == 1] = 'First'

test_data.Pclass[test_data.Pclass == 2] = 'Second'

test_data.Pclass[test_data.Pclass == 3] = 'Third'
# Lets create the dummy variables now

train_dummy = pd.get_dummies(train_data)
test_dummy = pd.get_dummies(test_data)
train_dummy.info()
print('\n--------------------\n')
test_dummy.info()
train_dummy.drop(['Pclass_First','Sex_male','Embarked_C'], axis =1, inplace = True)
test_dummy.drop(['Pclass_First','Sex_male','Embarked_C','PassengerId'], axis = 1, inplace = True)
Train_Y = train_dummy['Survived']
Train_X = train_dummy.drop(['Survived'], axis = 1)
Test_X = test_dummy
# Logistic Regression Model

LogReg = LogisticRegression()
LogReg.fit(Train_X, Train_Y)
Y_Pred = LogReg.predict(Test_X)
LogReg.score(Train_X, Train_Y)
# Support Vector Machines

svc = SVC()
svc.fit(Train_X, Train_Y)
Y_pred_svm = svc.predict(Test_X)
svc.score(Train_X, Train_Y)
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(Train_X, Train_Y)

Y_pred_rf = random_forest.predict(Test_X)

random_forest.score(Train_X, Train_Y)
# Using KNN

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(Train_X, Train_Y)

Y_pred_Knn = knn.predict(Test_X)

knn.score(Train_X, Train_Y)
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(Train_X, Train_Y)

Y_pred_GNB = gaussian.predict(Test_X)

gaussian.score(Train_X, Train_Y)
Y = pd.DataFrame()
Y['Logistic'] = Y_Pred
Y['SVM'] = Y_pred_svm
Y['RF'] = Y_pred_rf
Y['KNN'] = Y_pred_Knn
Y['GNB'] = Y_pred_GNB
# Now lets see the correlation between all these models
Y.corr()
Final_Pred = (Y_pred_rf + Y_pred_Knn + Y_pred_svm )/3
Final_Pred = np.round(Final_Pred)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Final_Pred
    })

submission.to_csv('titanic.csv', index=False)