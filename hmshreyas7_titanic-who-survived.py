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
titanic_data = pd.read_csv('../input/train.csv')

titanic_data.head()
titanic_data.describe()
feature_columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_X = titanic_data.loc[:, feature_columns]
train_y = titanic_data.Survived
train_X = pd.get_dummies(train_X)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
train_X = pd.DataFrame(data = imputer.fit_transform(train_X), columns = ['Pclass','Age','SibSp','Parch','Fare','Sex_Female','Sex_Male','Embarked_C','Embarked_Q','Embarked_S'])

train_X.head()
from sklearn import svm
model = svm.SVC(kernel = 'linear', random_state = 1, gamma = 'scale', C = 100)
model.fit(train_X, train_y)
test = pd.read_csv('../input/test.csv')
test_X = test.loc[:, feature_columns]
test_X = pd.get_dummies(test_X)
test_X = pd.DataFrame(data = imputer.transform(test_X), columns = ['Pclass','Age','SibSp','Parch','Fare','Sex_Female','Sex_Male','Embarked_C','Embarked_Q','Embarked_S'])

test_X.head()
predictions = model.predict(test_X)
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
submission.to_csv('submission.csv', index = False)