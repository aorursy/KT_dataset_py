import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Inspecting train file
train_orig = pd.read_csv('../input/train.csv')
test_orig=pd.read_csv('../input/test.csv')
print(test_orig.columns)
print(train_orig.columns)
print(train_orig.describe())
print(train_orig.head())
train=train_orig.copy()
print(train.head())
print(train.isnull().sum())
data_to_drop=['Name','Ticket','Fare','Cabin','Embarked']
train.drop(data_to_drop,axis=1,inplace=True)
print(train.columns)
y=train.Survived
train.drop('Survived',axis=1,inplace=True)
print(train.head())
print(train.columns)
print(train.isnull().sum())
print(y.head())

#Fill age column
train['Age'].fillna(train['Age'].mode()[0],inplace=True)
print(train.isnull().sum())

#Encoding the data
le_sex=LabelEncoder()
train['Sex']=le_sex.fit_transform(train['Sex'])
le_class=LabelEncoder()
train['Pclass']=le_sex.fit_transform(train['Pclass'])
print(train.columns)
ohe=OneHotEncoder(categorical_features=[1])
train=ohe.fit_transform(train)
#Splitting the data
x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.2,random_state=0)
#Use classifiers
model=RandomForestClassifier(n_estimators=140)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
score=accuracy_score(y_test,y_pred)
print("Accuracy score:",score*100)

#Inspecting test file
test=test_orig.copy()
print(test.columns)
print(test.isnull().sum())
test.drop(data_to_drop,axis=1,inplace=True)
print(test.columns)
test['Age'].fillna(test['Age'].mode()[0],inplace=True)
print(test.isnull().sum())

le_sex1=LabelEncoder()
test['Sex']=le_sex1.fit_transform(test['Sex'])
ohe1=OneHotEncoder(categorical_features=[1])
test=ohe1.fit_transform(test)

#Predicting the test file
predictions=model.predict(test)
#Submission
sub_path=pd.read_csv('../input/gender_submission.csv')
my_submission=pd.DataFrame({'PassengerId':sub_path['PassengerId'],'Survived':predictions})
my_submission.to_csv('gender_submission.csv',index=False)
