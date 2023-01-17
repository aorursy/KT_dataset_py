import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
# check the shape of the data
df.shape
# independent variable
X=df.drop(['Unnamed: 32','id','diagnosis'],axis=1)
X.head()
# Dependent Variable
y=df.diagnosis
y
# library
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
y=label.fit_transform(y)
y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99)

# checking the shape of the data
X_train.shape,X_test.shape,y_train.shape,y_test.shape
X_train
# import library
from sklearn.preprocessing import StandardScaler
# scale
scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)
X_train, X_test
# shape
X_train.shape, X_test.shape
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
# check the shape again
X_train.shape, X_test.shape
# Import the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
# model
model=Sequential()
# layers
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
# checking the summary
model.summary()
# compiling the model
model.compile(optimizer=Adam(learning_rate=0.00005),loss='binary_crossentropy',metrics=['accuracy'])
%%time
#fitting the model
history=model.fit(X_train,y_train,epochs=50, validation_data=(X_test,y_test))
# plotting
pd.DataFrame(history.history).plot(figsize=(10,8))
plt.grid(True)
plt.show()
