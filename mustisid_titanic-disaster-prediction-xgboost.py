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
titanic_data_train = pd.read_csv("/kaggle/input/titanic/train.csv") 
#preproccesing data
#1. removing some columns[PassengerId,Name,Ticket,Cabin]
titanic_data_train = titanic_data_train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

#2.Encoding text data['Sex','Embarked']
#Embarked C=0,Q=1 and S=2
for i in range(0,len(titanic_data_train)):
    if titanic_data_train['Embarked'][i] == 'C':
        titanic_data_train['Embarked'][i]=0
    if titanic_data_train['Embarked'][i] == 'Q':
        titanic_data_train['Embarked'][i]=1
    else:
        titanic_data_train['Embarked'][i]=2
#Sex male=0,female=1
for i in range(len(titanic_data_train)):
    if titanic_data_train['Sex'][i]=='male':
        titanic_data_train['Sex'][i] = 0
    else:
        titanic_data_train['Sex'][i]=1
        
#3.checking if any column has Nan value
for col in titanic_data_train.columns:
    is_NaN = titanic_data_train[col].isnull().values.any()
    if is_NaN == True:
        avg= titanic_data_train[col].sum()//len(titanic_data_train)
        titanic_data_train[col]=titanic_data_train[col].fillna(avg)

#only if using XGBoost
titanic_data_train['Sex']=titanic_data_train['Sex'].astype('int')
titanic_data_train['Embarked'] = titanic_data_train['Embarked'].astype('int')

X = titanic_data_train.drop(['Survived'],axis=1)
Y = titanic_data_train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
titanic_data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
#preproccesing data 
#1. removing some columns[PassengerId,Name,Ticket,Cabin]
titanic_data_test_edit = titanic_data_test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

#2.Encoding text data['Sex','Embarked']
#Embarked C=0,Q=1 and S=2
for i in range(0,len(titanic_data_test_edit)):
    if titanic_data_test_edit['Embarked'][i] == 'C':
        titanic_data_test_edit['Embarked'][i]=0
    if titanic_data_test_edit['Embarked'][i] == 'Q':
        titanic_data_test_edit['Embarked'][i]=1
    else:
        titanic_data_test_edit['Embarked'][i]=2
#Sex male=0,female=1
for i in range(len(titanic_data_test_edit)):
    if titanic_data_test_edit['Sex'][i]=='male':
        titanic_data_test_edit['Sex'][i] = 0
    else:
        titanic_data_test_edit['Sex'][i]=1
        
#3.checking if any column has Nan value
for col in titanic_data_test_edit.columns:
    is_NaN = titanic_data_test_edit[col].isnull().values.any()
    if is_NaN == True:
        avg= titanic_data_test_edit[col].sum()//len(titanic_data_test_edit)
        titanic_data_test_edit[col]=titanic_data_test_edit[col].fillna(avg)
titanic_data_test_edit["Embarked"] = titanic_data_test_edit["Embarked"].astype('int')
titanic_data_test_edit["Sex"] = titanic_data_test_edit["Sex"].astype('int')
#using XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=3,learning_rate=0.001,n_estimators=100,booster='dart')
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred))
print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
test = classifier.predict(titanic_data_test_edit)
titanic_data_test['Survived'] = test

submission = titanic_data_test[['PassengerId','Survived']]
submission
submission.to_csv('kaggle\\working\\submission_test.csv',index=False)