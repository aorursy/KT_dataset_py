import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sea
from keras.datasets.mnist import load_data

from keras.layers import Dense,Embedding,Conv2D,MaxPooling2D,Flatten,Dropout

from keras.models import Sequential

from keras.losses import binary_crossentropy

from keras.optimizers import SGD,rmsprop,RMSprop

from keras.metrics import binary_accuracy

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
trainDataGen=ImageDataGenerator(rescale=1./255)

testDataGen=ImageDataGenerator(rescale=1./255)
trainData=trainDataGen.flow_from_directory("/kaggle/input/cat-and-dog/training_set/training_set",batch_size=20,class_mode='binary')

testData=testDataGen.flow_from_directory("/kaggle/input/cat-and-dog/test_set/test_set",batch_size=20,class_mode='binary')
trainData.image_shape
Y_train=to_categorical(trainData.classes)

Y_test=to_categorical(testData.classes)
Y_train.shape
sea.countplot(trainData.classes)
sea.countplot(testData.classes)
network=Sequential()

network.add(Conv2D(128,(3,3),activation='relu',input_shape=(256,256,3)))

network.add(MaxPooling2D((2,2)))

network.add(Conv2D(128,(3,3),activation='relu'))

network.add(MaxPooling2D((2,2)))

network.add(Conv2D(128,(3,3),activation='relu'))

network.add(MaxPooling2D((2,2)))

network.add(Conv2D(128,(3,3),activation='relu'))

network.add(MaxPooling2D((2,2)))

network.add(Conv2D(128,(3,3),activation='relu'))

network.add(Flatten())

network.add(Dropout(0.9))

network.add(Dense(256,activation='relu'))

network.add(Dropout(0.5))

network.add(Dense(256,activation='relu'))

network.add(Dropout(0.5))

network.add(Dense(256,activation='relu'))

network.add(Dense(1,activation='sigmoid'))
network.summary()
network.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.0001),metrics=['acc'])
network.fit_generator(trainData,steps_per_epoch=250,epochs=40)
network.evaluate_generator(testData,steps=250)
filename="/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1.jpg"

plt.imshow(plt.imread(filename))
from keras.preprocessing import image



img=image.load_img(filename,target_size=(256,256))

imgArray=image.img_to_array(img)

imgTensor=np.expand_dims(imgArray,axis=0)/255
imgTensor.shape
from keras import models

layer_output=[layer.output for layer in network.layers[:4]]

activationModel=models.Model(inputs=network.input,output=layer_output)
activation=activationModel.predict(imgTensor)[2]

activation.shape
plt.matshow(activation[0,:,:,4])
plt.matshow(activation[0,:,:,50])