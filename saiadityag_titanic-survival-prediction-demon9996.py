# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv('../input/test.csv')
submission_sample=pd.read_csv("../input/gender_submission.csv")
train.head()
test.head()
train.isnull().sum()

train[['Age','Cabin','Embarked']] = train[['Age','Cabin','Embarked']].replace(0, np.NaN)
# fill missing values with mean column values
train.fillna(train.mean(), inplace=True)
# count the number of NaN values in each column
train=train.drop('Cabin',axis=1)
train.dropna(inplace=True)
print(train.isnull().sum())
test[['Age','Cabin','Fare']] = test[['Age','Cabin','Fare']].replace(0, np.NaN)
test=test.drop(['Cabin','Name','Ticket'],axis=1)
# fill missing values with mean column values
test.fillna(test.mean(), inplace=True)
# count the number of NaN values in each column
test.dropna(inplace=True)
print(test.isnull().sum())

train1,validate = train_test_split(train,test_size = 0.3,random_state = 100)
train_y=train1['Survived']
train_x=train1.drop(['Survived','PassengerId','Name','Ticket'],axis=1)
train_x=pd.get_dummies(train_x)
test_x=test.drop(['PassengerId'],axis=1)
validate_y = validate['Survived']
validate_x = validate.drop(['Survived','PassengerId','Name','Ticket'],axis = 1)
validate_x = pd.get_dummies(validate_x)
model=DecisionTreeClassifier()
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
model=RandomForestClassifier(n_estimators=300)
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))

test_x=pd.get_dummies(test_x)
test_x.columns
model=LogisticRegression(solver='liblinear',C=1)
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test_x)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test_x)
from sklearn.svm import SVC  
model = SVC(kernel='linear')
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test_x)
df_test_predict = pd.DataFrame(test_pred,columns = ['Survived'])
df_test_predict['PassengerId'] = test['PassengerId']
df_test_predict[['PassengerId','Survived']].to_csv('submission_1.csv',index = False)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=300)
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
#print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test_x)
test_pred=np.round(test_pred)
test_pred=test_pred.astype(int)
print(test_pred)
df_test_predict = pd.DataFrame(test_pred,columns = ['Survived'])
df_test_predict['PassengerId'] = test['PassengerId']
df_test_predict[['PassengerId','Survived']].to_csv('submission_1.csv',index = False)