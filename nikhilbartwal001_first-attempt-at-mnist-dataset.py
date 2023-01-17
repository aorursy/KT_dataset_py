#Importing necessary libraries

import numpy as np

import seaborn as sns

import pandas as pd

import os

import re

import matplotlib.pyplot as plt

%matplotlib inline
os.listdir("../input/digit-recognizer")
#Loading the training and test data

train_data = pd.read_csv("../input/digit-recognizer/train.csv")

test_data = pd.read_csv("../input/digit-recognizer/test.csv")

sample_submission_data = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train_data.head()
x_train = train_data.drop('label',axis=1)

y_train = train_data['label'].astype('int32')



print("X train shape: " + str(x_train.shape))

print("Y train shape: " + str(y_train.shape))

print("X test shape: " + str(test_data.shape))
print("Number of images per digit:\n" + str(y_train.value_counts()))
sns.countplot(y_train)
#Normalizing pixel values

x_train /= 255

test_data /= 255



x_train = x_train.values.reshape(x_train.shape[0],28,28,1)

test_data = test_data.values.reshape(test_data.shape[0],28,28,1)



print("X train shape: " + str(x_train.shape))

print("Test data shape: " + str(test_data.shape))
#Plotting the 10 starting images

numIm = 10

plt.figure(figsize=(12,10))

for img in range(numIm):

    plt.subplot(numIm/5 + 3,5,img+1)

    plt.imshow(x_train[img].reshape(28,28),cmap='binary_r')

    plt.axis("off")

    plt.title('Label: ' + y_train[img].astype('str'))

print(type(x_train))

print(y_train.shape)
#Importing libraries for creating the deep NN model

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,AveragePooling2D

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
#Splitting the training data into train and test(cross-validation) set

y_train = to_categorical(y_train,10)

x_train,x_test,y_train,y_test =  train_test_split(x_train, y_train,test_size=0.1,random_state=1)

 

print("X train " + str(x_train.shape))

print("Y train " + str(y_train.shape))

print("X test " + str(x_test.shape))

print("Y test " + str(y_test.shape))
#Creating the deeep CNN model

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32,(3,3),activation='relu'))



model.add(Dropout(0.25))



model.add(Conv2D(16,(3,3),activation='relu'))



model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(400,activation='relu'))

model.add(Dense(75,activation='relu'))



model.add(Dropout(0.25))



model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
#Training the final model

model.fit(x_train,y_train,batch_size=128,epochs=10,verbose=1,validation_data = (x_test,y_test))
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

train_stat = model.evaluate(x_train, y_train, verbose=0)

print('Train Loss:     ', round(train_stat[0], 5))

print('Train Accuracy: ', round(train_stat[1]*100, 4), '%')

print('Test Loss:      ', round(loss, 5))

print('Test Accuracy:  ', round(accuracy*100, 4), '%')
predictions = model.predict(test_data)
#Storing the results in a csv file for submissions

results = np.argmax(predictions,axis=1)

results = pd.Series(results,name='Label')

submission = pd.concat([pd.Series(range(1,28001),name='ImageID'),results],axis=1)

submission.to_csv("submission.csv",index=False)