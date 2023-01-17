import keras

import pandas as pd

import numpy as np

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam
labels=[None]*6000

id = [None]*6000

j=0

k=0

for file in os.listdir("../input/socofing/socofing/SOCOFing/Real"):

    id[k] = file

    k+=1

    for i in range (len(file)):

        if(file[i]=="_"):

            if file[i+2]=="M":

                labels[j]=0

            else:

                labels[j]=1

            j=j+1

            break
import keras_applications

model  = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
print("hi")

#print(models.get_config())
from keras.layers import Dense, Activation, Flatten

base_model=model

x=base_model.output

print(x.shape)

#x = GlobalAveragePooling2D()(x)

print(x.shape)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

print(x.shape)

x=Dense(1024,activation='relu')(x) #dense layer 2

print(x.shape)

x=Dense(512,activation='relu')(x) #dense layer 3

print(x.shape)

x=Dense(128,activation='relu')(x) #dense layer 3

print(x.shape)

x=Dense(32,activation='relu')(x) #dense layer 3

print(x.shape)

preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

print(x.shape,preds)
model=Model(inputs=base_model.input,outputs=preds)

#trainset = ImageDataset(csv_file = id, root_dir = '../input/socofing/socofing/SOCOFing/Real/', labels = labels ,transform=transforma)

#valset = ImageDataset(csv_file = id, root_dir = '../input/socofing/socofing/SOCOFing/Real/', labels = labels ,transform=transformb)
print(len(model.layers))
for layer in model.layers:

    layer.trainable=False

for layer in model.layers:

    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

tenlabels = pd.DataFrame(labels)

train_df = list(zip(labels,id))

train_Df = pd.DataFrame(train_df,columns=["class","filename"],index = None)



train_generator=train_datagen.flow_from_dataframe(dataframe = train_Df,directory = '../input/socofing/socofing/SOCOFing/Real/', # this is where you specify the path to the main data folder

                                                 xcol = "filename",

                                                 ycol = "class",

                                                 target_size=(224,224),

                                                 color_mode='rgb',

                                                 batch_size=16,

                                                 class_mode='other',

                                                 shuffle=True)
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size

model.fit_generator(generator=train_generator,

                   steps_per_epoch=step_size_train,

                   epochs=3)
model.save("my_model.h5")
import tensorflow as tf

new_model= tf.keras.models.load_model(filepath="my_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model_file("my_model.h5")

tflite_model = converter.convert()

open("mobileNet_model.tflite", "wb").write(tflite_model)