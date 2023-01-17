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
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

y_train = train_data.Survived
X_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
# replace non numeric variables by numeric ones
one_hot_encoded_X_train = pd.get_dummies(X_train)
one_hot_encoded_X_test = pd.get_dummies(X_test)

#aligning train and test predictors
X_train, X_test = one_hot_encoded_X_train.align(one_hot_encoded_X_test, join='left', axis=1)
#Handling Missing Values by imputing
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(one_hot_encoded_X_train))
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(one_hot_encoded_X_test))
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
imputed_X_train = sc.fit_transform(imputed_X_train)
imputed_X_test = sc.transform(imputed_X_test)
from sklearn.svm import SVC

my_classifier = SVC(kernel = 'rbf', random_state = 0)
my_classifier.fit(imputed_X_train, y_train)
y_pred = pd.DataFrame(my_classifier.predict(imputed_X_test))

#Creating PassengerID column
e =[]
for num in range(892, 1310):
    e.append(num)
# adding the new column to y_pred DataFrame
y_pred['e'] = e

#adding headers after being deleted during imputation
y_pred.columns=['Survived', 'PassengerId']

#swiching colums to the right order to match the needed output formula
y_pred = y_pred[['PassengerId', 'Survived']]
y_pred.to_csv('Titania_Kernel_SVM.csv', index=False)
