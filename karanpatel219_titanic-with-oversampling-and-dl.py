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
import pandas as pd 
import matplotlib.pyplot as plt
train=pd.read_csv('../input/titanic/train.csv')
train
train.drop(['Name','Cabin','PassengerId'],axis=1,inplace=True)
test=pd.read_csv('../input/titanic/test.csv')
test.drop(['Name','Cabin','PassengerId'],axis=1,inplace=True)
train.isnull().sum()
test['Age'].isnull().sum()
train['fam']=train['SibSp']+train['Parch']
test['fam']=test['SibSp']+test['Parch']
import matplotlib.pyplot as plt
plt.hist(train[train['Survived']==1]['Age'],bins=[0,10,20,30,40,50,60,70,80,90],label='Survived',alpha=0.8)
plt.hist(train[train['Survived']==0]['Age'],bins=[0,10,20,30,40,50,60,70,80,90],alpha=0.75,label="Not Survived")
plt.xlabel('Age Distribution')
plt.ylabel('Passengers')
plt.legend()
train[train['Survived']==1]
plt.hist(train[train['Survived']==1]['Survived'],bins=[0,0.25,0.5,0.75,1],label='Survived')
plt.hist(train[train['Survived']==0]['Survived'],bins=[0,0.25,0.5,0.75,1],label='Not Survived')
plt.xlabel('Survived')
plt.ylabel('Passengers')
plt.legend()
plt.hist(train[train['Survived']==1]['Pclass'],bins=[1,1.5,2,2.5,3,3.5,4],label='Survived',alpha=0.8)
plt.hist(train[train['Survived']==0]['Pclass'],bins=[1,1.5,2,2.5,3,3.5,4],alpha=0.5,label='Not Survived')
plt.xlabel('Class')
plt.ylabel('Passengers')
plt.legend()
train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
train
train.drop(['SibSp','Parch'],axis=1,inplace=True)
test.drop(['SibSp','Parch'],axis=1,inplace=True)
train.isnull().sum()
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
train['Embarked'].fillna('S',inplace=True)
test['Embarked'].fillna('S',inplace=True)
test.isnull().sum()
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
Y=train['Survived']
X=train.drop('Survived',axis=1)
X['Sex'].replace({'male':1,'female':0},inplace=True)
test['Sex'].replace({'male':1,'female':0},inplace=True)
from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder()
X['Embarked']=l1.fit_transform(X['Embarked'])
test['Embarked']=l1.fit_transform(test['Embarked'])
X
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
test=sc.fit_transform(test)
from imblearn.over_sampling import SMOTE
os=SMOTE()
X_os,Y_os=os.fit_resample(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
Xos_train,Xos_test,Yos_train,Yos_test=train_test_split(X_os,Y_os)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
model1=LogisticRegression()
model1.fit(X_train,Y_train)
print(model1.score(X_train,Y_train))

model2=LogisticRegression()
model2.fit(Xos_train,Yos_train)
print(model2.score(Xos_train,Yos_train))

model3=RandomForestClassifier()
model3.fit(X_train,Y_train)
print(model3.score(X_train,Y_train))

ypr=model3.predict(X_test)
from sklearn.metrics import f1_score,confusion_matrix
print(confusion_matrix(ypr,Y_test))
print(f1_score(ypr,Y_test))
model5=SVC()
model5.fit(X_train,Y_train)
print(model5.score(X_train,Y_train))
ypr1=model5.predict(X_test)
print(confusion_matrix(ypr1,Y_test))
print(f1_score(ypr1,Y_test))
model6=SVC()
model6.fit(Xos_train,Yos_train)
print(model6.score(Xos_train,Yos_train))
ypr2=model6.predict(Xos_test)
print(confusion_matrix(ypr2,Yos_test))
print(f1_score(ypr2,Yos_test))
ypr2=model6.predict(test)
sub=pd.read_csv('../input/titanic/gender_submission.csv')

sub['Survived']=ypr2
sub
sub.to_csv('sub3.csv',index=False)
from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()
model.add(Dense(5,'relu',input_shape=(6,)))
model.add(Dense(16,'relu'))
model.add(Dense(32,'relu'))
model.add(Dense(64,'relu'))
model.add(Dropout(0.1))
model.add(Dense(64,'relu'))
model.add(Dropout(0.1))
model.add(Dropout(0.1))
model.add(Dense(16,'relu'))
model.add(Dense(8,'relu'))
model.add(Dense(1,'sigmoid'))
model.compile('adam','binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=100,batch_size=5)
model.evaluate(X_test,Y_test)
model.evaluate(Xos_test,Yos_test)
yp=model.predict(test)
sub['Survived']=yp
sub['Survived']=round(sub['Survived'])
sub['Survived']=sub['Survived'].astype('int64')
sub.to_csv('subDL1.csv',index=False)