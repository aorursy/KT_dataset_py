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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.shape)
print(test.shape)
test.head()
train.head()
train.isnull().sum()
test.isnull().sum()
train.drop(columns=['Cabin'],axis=1,inplace=True)
test.drop(columns=['Cabin'],axis=1,inplace=True)
test.head()
# 1. Fill missing values of fare col(test)
# 2. Fill missing values of age(both train and test)
# 3. Fill missing values of embarked(train)
# Fare missing value

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
test.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].fillna('S')
age_train = np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std(),177)
age_test = np.random.randint(test['Age'].mean()-test['Age'].std(),test['Age'].mean()+test['Age'].std(),86)
train['Age'][train['Age'].isnull()] = age_train
test['Age'][test['Age'].isnull()] = age_test
train['family']=train['SibSp'] + train['Parch'] + 1
test['family']=test['SibSp'] + test['Parch'] + 1
def family(number):
    
    if number == 1:
        return "Alone"
    elif number >1 and number <=4:
        return "Small"
    else:
        return "Large"
train['family_type'] = train['family'].apply(family)
test['family_type'] = test['family'].apply(family)
train.drop(columns=['SibSp','Parch','family'],axis=1,inplace=True)
test.drop(columns=['SibSp','Parch','family'],axis=1,inplace=True)
passengerId = test['PassengerId']
train.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
test.drop(columns=['PassengerId','Name','Ticket'],axis=1,inplace=True)
train = pd.get_dummies(train,columns=['Pclass','Sex','family_type','Embarked'],drop_first=True)
test = pd.get_dummies(test,columns=['Pclass','Sex','family_type','Embarked'],drop_first=True)
X = train.iloc[:,1:].values
y = train.iloc[:, 0].values
X_pred = test.iloc[:,:].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)
print(X_train.shape)
print(X_test.shape)
# Algorithm apply
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
y_final = clf.predict(X_pred)
submission = pd.DataFrame()
submission['passengerId']=passengerId
submission['Survived']=y_final
submission.head()
submission.to_csv('my_submission.csv',index=False)
