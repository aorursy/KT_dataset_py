# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#explore the data a little bit
print(train_data.columns.values)
print(train_data.describe())
train_data.head()
#let's turn sex into a numerical feature instead of categorical
from sklearn.preprocessing import LabelEncoder
train_data['Sex'] = LabelEncoder().fit_transform(train_data['Sex'])
print(train_data.isnull().sum())
abc = train_data['Age']
print(abc.shape)
#handling missing values
#print(train_data.isnull().sum())
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
train_data['Age'] = imp.fit_transform(train_data['Age'].values.reshape(-1,1)).reshape(-1)
print(train_data.isnull().sum())
train_data['Age'].mean()
y = train_data.Survived
predictors = ['Pclass','Sex','Age','SibSp','Parch','Fare']
X = train_data[predictors]
#alright let's fit some models and see how it goes
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

DT_model = DecisionTreeClassifier()
DT_model.fit(train_X,train_y)

RF_model = RandomForestClassifier()
RF_model.fit(train_X,train_y)

KNN_model = KNeighborsClassifier()
KNN_model.fit(train_X,train_y)

DT_predictions = DT_model.predict(val_X)
RF_predictions = RF_model.predict(val_X)
KNN_predictions = KNN_model.predict(val_X)

print("Decision tree accuracy: ", accuracy_score(val_y,DT_predictions))
print("Random forest accuracy: ", accuracy_score(val_y,RF_predictions))
print("KNN accuracy: ", accuracy_score(val_y,KNN_predictions))
print(test_data.isnull().sum())
#output data
test_data['Sex'] = LabelEncoder().fit_transform(test_data['Sex'])
test_data['Age'] = imp.fit_transform(test_data['Age'].values.reshape(-1,1)).reshape(-1)
test_data['Fare'] = imp.fit_transform(test_data['Fare'].values.reshape(-1,1)).reshape(-1)
test_X = test_data[predictors]

print(test_X.isnull().sum())

DT_predictions = DT_model.predict(test_X)
RF_predictions = RF_model.predict(test_X)
KNN_predictions = KNN_model.predict(test_X)
DT_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': DT_predictions})
DT_submission.head()
DT_submission.to_csv('DT-submission.csv', index=False)
RF_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': RF_predictions})
RF_submission.head()
RF_submission.to_csv('RF-submission.csv', index=False)
KNN_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': KNN_predictions})
KNN_submission.head()
KNN_submission.to_csv('KNN-submission.csv', index=False)
