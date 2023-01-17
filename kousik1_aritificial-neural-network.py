import numpy as np

import pandas as pd

import tensorflow as tf
#Data Preprocessing

dataset=pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values

y=dataset.iloc[:,-1].values
print(x)
print(y)
#Encoding the Categorical Data

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()

x[:,2]=LE.fit_transform(x[:,2])
x
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

CT=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder="passthrough")

x=np.array(CT.fit_transform(x))

print(x)
#Splitting the into training set and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
x_train
x_test
#Initializing the ANN

ann=tf.keras.models.Sequential()
#Adding the layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6,activation="relu"))
#Adding the output layer

ann.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
#Compiling the ann

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Training the ANN on the training set

ann.fit(x_train,y_train,batch_size=32,epochs=100)
#Making the predictions and evaluating model

y_pred=ann.predict(x_test)

y_pred=(y_pred>0.6)

print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#Making the confusion matrix

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)

print(cm)

accuracy_score(y_test,y_pred)