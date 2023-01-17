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
# load dataset

dataset_train = pd.read_csv('../input/train.csv')

dataset_test = pd.read_csv('../input/test.csv')

print(dataset_train.head())

print(dataset_train.shape)

print(dataset_train.columns)

print(dataset_test.columns)
X_train = dataset_train.loc[:, ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

y_train = dataset_train.loc[:, ['Survived']]

X_test = dataset_test.loc[:, ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]



X_train['Age'].median()

X_train['Age'].isnull().sum()

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())

X_train['Age'].isnull().sum()

X_test['Age'].isnull().sum()

X_test['Age'] = X_test['Age'].fillna(X_test['Age'].median())

X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].median())



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)



out_data = pd.DataFrame({'Passenger_ID' : X_test['PassengerId'],

                         'Survived' : y_pred })



out_data.to_csv("Final_submission.csv", index=False)

print(out_data.head())
