# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Important libraries for working out with images preprocessing and training

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import warnings

warnings.filterwarnings("ignore")

import keras

from keras.layers import Dense,Convolution2D,Dropout,MaxPooling2D,BatchNormalization,Flatten

from keras.models import Sequential,Model

from keras.preprocessing.image import ImageDataGenerator

# Looking for the image directories

os.listdir("../input/repository/shobhitsrivastava-ds-Violence-a245c62/Images/")
# Setting the image path

path = "../input/repository/shobhitsrivastava-ds-Violence-a245c62/Images/"
#Getting data generated from the directories through image data generator

data = ImageDataGenerator(rescale = 1./255, zoom_range = 0.3,horizontal_flip=True,rotation_range= 15).flow_from_directory(path,target_size= (224,224),color_mode= "rgb",classes= ["Rifle","tank","guns","knife images"],batch_size=90)
x,y = data.next()

plt.subplot(4,3,2)

for i in range(0,12):

    image = x[i]

    label = y[i]

    print (label)

    plt.imshow(image)

    plt.show()
len(data)
# Defining the Sequential model

model= Sequential()
#Adding up the layers of the network

model.add(Convolution2D(32,(3,3),input_shape=(224,224,3),padding = "Same",activation="relu"))

model.add(MaxPooling2D(pool_size=(3,3)))

#model.add(Dropout(0.2))

model.add(Convolution2D(32,(3,3),padding = "Same",activation="relu"))

model.add(MaxPooling2D(pool_size=(3,3)))

#model.add(Dropout(0.2))

model.add(Convolution2D(64,(3,3),padding = "Same",activation="relu"))

model.add(MaxPooling2D(pool_size=(3,3)))

#model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.3))

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(4,activation="softmax"))
# Implementing the callback function so as to stop the algorithm from the furthur traning in case the accuracy dips down

clbk= keras.callbacks.EarlyStopping(monitor='accuracy',mode='min')
#Cmpiling the model

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Printing out the model summary

model.summary()
# Training the model  

history_1 = model.fit_generator(data,steps_per_epoch=int(1273/20),epochs=10,shuffle=False,callbacks=[clbk])
data
history_1.history
model.save("Mymodel_2.h5")
loss= history_1.history["loss"]

acc= history_1.history["acc"]
# Plotting the model loss

plt.plot(loss,color="r")

plt.title("Loss progression curve")
# Plotting the model accuracy

plt.plot(acc,color="b")

plt.title(" Accuracy progression curve")
#Importig the transfer learning model VGG19

from keras.applications import VGG19
# Assigning weight and input shape

model_sec=VGG19(weights="imagenet",include_top=False,input_shape=(224,224,3))

model_sec.summary()
# Generating the data

data_final = ImageDataGenerator(rescale = 1/255, zoom_range = 0.2,horizontal_flip=True,vertical_flip=True).flow_from_directory(path,target_size=(224,224),color_mode="rgb",classes=["Rifle","tank","guns","knife images"],batch_size=90)
# Making the strting top layers of the model as non-trainable

for layer in model_sec.layers:

    layer.trainable=False
model_2=model_sec.output
# Adding the last trainable layers to the model

model_2= Flatten()(model_2)

model_2= Dense(512,activation="relu")(model_2)

model_2= Dropout(0.3)(model_2)

model_2= Dense(256,activation="relu")(model_2)

model_2= Dropout(0.3)(model_2)

pred= Dense(4,activation="softmax")(model_2)

model_final =Model(input=model_sec.input,output=pred)
# Compiling the model

model_final.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
# Training

history = model_final.fit_generator(data_final,steps_per_epoch=int(1273/80),epochs=8,shuffle=False,callbacks=[clbk])
history.history
model_final.save("Myfinal_model_2.h5")
loss_final= history.history["loss"]

acc_final = history.history["acc"]
plt.plot(loss_final,color="r")

plt.title("Loss Progression Curve")
plt.plot(acc_final,color="b")

plt.title("Accuracy Progression Curve")