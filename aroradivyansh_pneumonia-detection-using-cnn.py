import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import PIL
from PIL import Image
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        '/kaggle/input/chest-xray-pneumonia/chest_xray/train/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
val_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
val_set = train_datagen.flow_from_directory(
        '/kaggle/input/chest-xray-pneumonia/chest_xray/val/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_set = train_datagen.flow_from_directory(
        '/kaggle/input/chest-xray-pneumonia/chest_xray/test/',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')
image="../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1011_bacteria_2942.jpeg"
PIL.Image.open(image)
image="../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0151-0001.jpeg"
PIL.Image.open(image)

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32 , kernel_size=3 ,
                               activation='relu',input_shape=[64,64,3]))
#features = no.of feature detectors
#kernelsize = size of feature det array
#input_shape = when we add first ip layer we specify shape ... 3 = rgb
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides = 2))
#pool size = size of matrix, stride = shift by no. of pixels 
cnn.add(tf.keras.layers.Conv2D(filters=32 , kernel_size=3 ,
                               activation='relu'))
#remove input_shape this is used to connect 1st layer to input layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
#128 hidden neurons
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
cnn.fit(x = training_set , validation_data=test_set , epochs = 30 )
print("Loss of the model is - " , cnn.evaluate(test_set)[0]*100 , "%")
print("Accuracy of the model is - " , cnn.evaluate(test_set)[1]*100 , "%")
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person100_bacteria_477.jpeg',target_size=(64,64)) 
#we change size to 64,64 same as in training and test set
#convert PIL into array as predict accept 2d array
test_image = image.img_to_array(test_image)
#in data pre. our network is trained in batches even if we apply predicion on single image we need to specify as image
test_image = np.expand_dims(test_image, axis=0)
#we are adding dim which is batch ... batch is first dim because first we have dim then in batch we have image color etc.. so axis-0
result = cnn.predict(test_image)
if result[0][0]== 1 :
  prediciton = 'PNEUMONIA'
else:
  predicion = 'NORMAL'
print(prediciton)
#WE TOOK PNEUMONIC IMAGE FROM TEST ... LET'S CHECK WHAT IT PREDICTS
cnn.save_weights("model.h5")
print("Saved model to disk")
