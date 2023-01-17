import pandas as pd

import numpy as np

train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
train_col =['Pclass','Sex','Age','SibSp']

target_col=['Survived']
trainX=train[train_col]

trainX.head()
trainY=train[target_col]

trainY.head()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
trainX['Sex'] = le.fit_transform(trainX['Sex'])

trainX

trainY.head()
trainX['Age'] = trainX['Age'].fillna(trainX['Age'].mean())
trainX.isnull().sum()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(trainX,trainY)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=2000)
model.fit(x_train,y_train)
model.score(x_train,y_train)
model.score(x_test,y_test)
FX = test[train_col]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

FX['Sex'] = le.fit_transform(FX['Sex'])
FX['Age'] = FX['Age'].fillna(FX['Age'].mean())
y_pred = model.predict(FX)
y_test.head()
y_pred
submit = pd.DataFrame(

 {

         'PassengerId':test.PassengerId,

         'Survived':y_pred 

     

 }

)

submit.to_csv('submition.csv',index=False)
test = pd.read_csv('../input/test.csv')

FX = test[train_col]
FX['Age'] = FX['Age'].fillna(FX['Age'].mean())
FX.isnull().sum()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

FX.head()
y_pred_final = model.predict(FX)