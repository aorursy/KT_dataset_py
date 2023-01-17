import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten
#Load test data and train data from the dataset
traindata = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
#seperate out input and output data
ytrain = traindata["label"]
xtrain = traindata.drop(labels = ["label"] , axis=1)
#convert data from 784 pixel values into image of 28*28
xtrain = xtrain.values.reshape(-1,28,28,1)
test = testdata.values.reshape(-1,28,28,1)
#normalize the data
xtrain=xtrain/255.0
testdata=testdata/255.0
#convert the labels into one-hot encoded vectors
ytrain = tf.keras.utils.to_categorical(ytrain,num_classes=10)
#create a model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(16,kernel_size=(3,3),activation = 'relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# for fitting
model.fit(xtrain,ytrain,batch_size=100,epochs=10,validation_split=0.2)
#prediction of test data using above model  
output = np.argmax(model.predict(test),axis = 1)
#convert the predicted data into a 1D array with name 'label'
output = pd.Series(output,name='label')
#concatenation of ImageId with predicted output
final_output = pd.concat([pd.Series(range(1,28001),name='ImageId'),output],axis=1)
#convert the array into csv file
final_output.to_csv("Handwritten-digit-recognition.csv",index=False)