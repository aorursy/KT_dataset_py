import tensorflow as tf
tf.test.gpu_device_name()
!pip install keras
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("../input/churn-modellingcsv/Churn_Modelling.csv")
df.head(5)
X=df.iloc[:,3:13]
y=df.iloc[:,13]
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)
X=pd.concat([X,geography,gender],axis=1)
X=X.drop(['Geography','Gender'],axis=1)
X.head(10)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
import keras
from keras.models import Sequential #for any kind od model either it be ANN or CNN or RNN
from Keras.layers import Dense #for any hidden layers that we will create
from keras.layers import Dropout #for using dropout function

import sys
!{sys. executable} -m pip install keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
classifier=Sequential()
#Adding the Input Layer and the first hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))
#Adding the 2nd hidden layer of 6 neuron
classifier.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu'))
#adding the output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
#Compiling the Ann model
classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.summary()
model_history=classifier.fit(X_train,y_train,validation_split=0.33,batch_size=10,epochs=50)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
cm
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)
score
