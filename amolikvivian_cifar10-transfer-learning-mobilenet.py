import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
#Loading dataset

(trainX, trainY), (testX, testY) = cifar10.load_data()
#Analyzing shape of data
print('Train ', '\nData :', trainX.shape, '\nLabel :', trainY.shape)

print('\nTest ', '\nData :', testX.shape, '\nLabel :', testY.shape)
#Visualizing image from training dataset

img = plt.imshow(trainX[1])
#One Hot Encoding Labels from Train and Test Dataset
from keras.utils import to_categorical

trainY_one_hot = to_categorical(trainY)
testY_one_hot = to_categorical(testY)
#Initializing MobileNet as Base Model for Transfer Learning

from keras.applications import MobileNet

base_model = MobileNet(include_top=False, weights='imagenet',
            input_shape=(32,32,3), classes=trainY.shape[1])
#Adding layers to base model of MobileNet

model = Sequential()

#Creating base layer of VGG19
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())

#Adding the Dense Layers and Dropout
model.add(Dense(512,activation=('relu'))) 

model.add(Dense(256,activation=('relu'))) 

model.add(Dropout(.3))

model.add(Dense(128,activation=('relu')))

model.add(Dropout(.2))

model.add(Dense(10,activation=('softmax')))
#Visualizing Model Summary
model.summary()
#Compiling Model using SGD 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
#Training Model
hist = model.fit(trainX, trainY_one_hot, batch_size = 100, epochs = 20, 
                 validation_split = 0.1)
#Testing accuracy of trained model

model.evaluate(testX, testY_one_hot)[1]
#Visualizing Model Accuracy

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc = 'upper left')