# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from IPython.display import clear_output

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D , Dense , Activation , Flatten , Dropout , MaxPooling2D

from keras.callbacks import BaseLogger, Callback

from sklearn.preprocessing import LabelBinarizer

from keras.optimizers import SGD

from keras.datasets import cifar10

import matplotlib.pyplot as plt

import os

import json

from livelossplot.keras import PlotLossesCallback
((trainX,trainY), (testX , testY)) = cifar10.load_data() 
labels = np.unique(trainY , return_counts= False)

list(labels)
trainX = trainX/255.0

testX = testX/255.0



lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)

testY = lb.transform(testY)
class MiniVGGnet:

    

    def build(image_size , classes):

        

        model = Sequential()

        model.add(Conv2D(32, (3,3) , padding='same', input_shape=image_size))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Conv2D(32, (3,3) , padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3) , padding='same'))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512))

        model.add(Activation('relu'))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))

        model.add(Dense(classes))

        model.add(Activation('softmax'))

        

        return model
class TrainingMonitor(Callback):

    

    def __init__(self):

        super(TrainingMonitor , self).__init__()

        

        

    def on_train_begin(self , logs={}):

        

        self.logs=[]

        self.losses=[]

        self.val_losses=[]

        self.acc=[]

        self.val_acc=[]

                        

    def on_train_end(self,epoch , logs={}):

        

        self.logs.append(logs)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        

        

        if len(self.losses)>1:

            

            clear_output(wait=True)

            N = np.arange(0,len(self.losses))

            plt.style.use('seaborn')

            plt.figure()

            plt.plot(N,self.losses , label='train_loss')

            plt.plot(N, self.val_losses , label='val_loss')

            plt.plot(N, self.acc , label='acc')

            plt.plot(N, self.val_acc , label='val_acc')

            plt.title('Training Loss and Accuracy [Epoch {}]'.format(len(self.losses)))

            plt.xlabel('Epoch #')

            plt.ylabel('Loss/Accuracy')

            plt.legend()

            plt.show()

            #plt.save("../working/epoch_{}.png".format(epoch))

            
print('[INFO] compiling model.....')

opt = SGD(lr=0.01 , momentum=0.9 , nesterov=True)

model = MiniVGGnet.build(image_size=(32,32,3) , classes=10)

model.compile(loss='categorical_crossentropy' , optimizer=opt , metrics=['accuracy'])

callbacks= [TrainingMonitor()]
#train the network

print("[INFO] training network")

model.fit(trainX, trainY ,

          validation_data=(testX , testY) , 

         epochs=5, 

         callbacks=[PlotLossesCallback()])
from keras.callbacks import TensorBoard

import time

tensor = TensorBoard(log_dir="../working/logs/{}".format(time.time()))
model.fit(trainX, trainY ,

          validation_data=(testX , testY) ,

          batch_size=64,

         epochs=5, 

         callbacks=[tensor])