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
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
train.isnull().sum()
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
train['Age'].fillna(train['Age'].median(), inplace = True)
train.drop(['Cabin'],axis=1,inplace=True)
train.isnull().sum().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(train['Survived'])
sns.countplot(x='Embarked',hue='Survived',data=train)
sns.countplot(x='Sex',hue='Survived',data=train)
train['Age'].hist()
sns.kdeplot(train[train['Survived']==0]['Age'],shade=True)
sns.kdeplot(train[train['Survived']==1]['Age'],shade=True)
plt.legend(['Not S','S'])
train['SibSp'].value_counts().plot(kind='bar')
sns.barplot(train['SibSp'],y='Survived',data=train)
train['Parch'].value_counts().plot(kind='bar')
sns.barplot(x=train['Parch'],y='Survived',data=train)
train['Fare'].hist()
train['Age'].describe()
def binning(x):
    if x>=0 and x<22:
        return 0
    elif x>=22 and x<28:
        return 1
    elif x>=28 and x<35:
        return 2
    else:
        return 3
train['Age']=train['Age'].map(binning)   
train['Fare']=train['Fare'].astype('int64')
train['Fare'].describe()
def bin(x):
    if x>=0 and x<7:
        return 0
    elif x>=7 and x<14:
        return 1
    elif x>=14 and x<31:
        return 2
    else:
        return 3
train['Fare']=train['Fare'].map(bin)   
from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
train['Sex']=lc.fit_transform(train['Sex'])
train['Embarked']=lc.fit_transform(train['Embarked'])
train.head()
Y=train['Survived']
X=train.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
input_dimension=X.shape[1]

model=Sequential()

model.add(Dense(4,input_dim=input_dimension,activation='relu',kernel_initializer="uniform"))
model.add(Dense(2,activation='relu',kernel_initializer="uniform"))
model.add(Dense(1,activation='sigmoid',kernel_initializer="uniform"))
#model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['Accuracy'])
model_history=model.fit(X_train,Y_train,batch_size=9,epochs=100,validation_data=(X_test,Y_test))
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,Y_test)
score