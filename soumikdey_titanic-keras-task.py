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
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df.shape
df.isnull().sum()
df=df.dropna()
X=df.drop('Survived',axis=1)
X.shape
y=df.Survived
y.shape
X.sample(5)
X.drop(['PassengerId','Ticket','Cabin','Name'],axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder
LR=LabelEncoder()
X['Sex']=LR.fit_transform(X['Sex'])
X['Embarked']=LR.fit_transform(X['Embarked'])
# AS It is possible that 2 gets more preference than 0 and 1
pd.get_dummies(X['Embarked'],drop_first=True)


X.sample(5)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit_transform(X)

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
X_train.shape
X_test.shape
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
model=Sequential()

model.add(Dense(12,activation='relu',input_dim=X_train.shape[1]))
model.add(Dense(12,activation='relu'))

model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=20,epochs=100,verbose=1,validation_split=0.2)
history.history
y_pred=model.predict_classes(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

#not overfitting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
