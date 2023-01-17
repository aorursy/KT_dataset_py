# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
combined = pd.concat([train.drop('Survived',axis=1),test])
train['Age'].fillna(train['Age'].median(),inplace=True)
train['Embarked'].fillna(train['Embarked'].value_counts().index[0], inplace=True) 
d = {1:'1st',2:'2nd',3:'3rd'} 
train['Pclass'] = train['Pclass'].map(d) 
train.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace=True) 
categorical_vars = train[['Pclass','Sex','Embarked']] 
dummies = pd.get_dummies(categorical_vars,drop_first=True)
train = train.drop(['Pclass','Sex','Embarked'],axis=1) 
train = pd.concat([train,dummies],axis=1) 
train.head() 

y = train['Survived']
X = train.drop(['Survived'],1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.1,max_depth=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
test['Age'].fillna(test['Age'].median(),inplace=True) 
test['Fare'].fillna(test['Fare'].median(),inplace=True)
d = {1:'1st',2:'2nd',3:'3rd'} 
test['Pclass'] = test['Pclass'].map(d)
test['Embarked'].fillna(test['Embarked'].value_counts().index[0], inplace=True)
ids = test[['PassengerId']]
test.drop(['PassengerId','Name','Ticket','Cabin'],1,inplace=True)
categorical_vars = test[['Pclass','Sex','Embarked']]
dummies = pd.get_dummies(categorical_vars,drop_first=True)
test = test.drop(['Pclass','Sex','Embarked'],axis=1)
test = pd.concat([test,dummies],axis=1)
preds = model.predict(test)
results = ids.assign(Survived=preds)
results.to_csv('titanic_submission.csv',index=False)
