import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D,Convolution2D


train= pd.read_csv('../input/digit-recognizer/train.csv')
test= pd.read_csv('../input/digit-recognizer/test.csv')
train.shape ,test.shape
train.head()
x_train=train.iloc[:,1:].values.astype('float32')
x_train
x_train.shape
y_train=train.iloc[:,0].values.astype('int32')
sns.countplot(y_train)
x_train = x_train.reshape(x_train.shape[0], 28, 28,1)
x_train.shape
test = test.values.astype('float32')
test = test.reshape(test.shape[0], 28, 28,1)
test.shape
#Lets normalize the data

x_train = x_train/255.0
test=test/255.0
x_train.shape
from keras.utils.np_utils import to_categorical
y_train= to_categorical(y_train)
#lets split the data

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.3,random_state=2)

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(x_val, y_val)
test_acc
from keras import layers
from keras import models
model1 = models.Sequential()
model1.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(128, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))


model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model1.fit(x_train, y_train, epochs=5, batch_size=64)
y_hat=model.predict(test)
y_pred = np.argmax(y_hat,axis=1)
results = pd.Series(y_pred,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("digits.csv",index=False)

submission
