import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import shutil

import random

import matplotlib.pyplot as plt # for ploting 

import cv2 # for reading images from folders

import tensorflow as tf # main framework for building nn

from tensorflow.keras import models, layers, optimizers, regularizers, utils

from sklearn.model_selection import train_test_split # for utility functions

from tensorflow.keras import applications

from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard

import time
CLASS_NUM = 10

folders_list = os.listdir("../input/stanford-dogs-dataset/images/Images")[0:CLASS_NUM]

path_list = [os.path.join('../input/stanford-dogs-dataset/images/Images', x) for x in folders_list]

path_list
breed_list = os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/') # list of all breeds for further demo

breed_list = [x.split('-')[1].capitalize() for x in breed_list][0:CLASS_NUM]

breed_list
#Reading images from folders into X with appropriate labels adding to Y

img_width, img_height, channels = 256, 256, 3

Y = []

X = []

for num,file in enumerate(path_list):

    for img in os.listdir(file):

        im = cv2.resize(cv2.imread(os.path.join(file, img)), (img_width, img_height)).astype(np.float32)

        X.append(im)

        Y.append(breed_list[num])

X = utils.normalize(X)
Y_num = []

Y_dict = dict(zip(set(Y), range(len(set(Y)))))

for elem in Y:

    Y_num.append(Y_dict[elem])

    

Y = Y_num     
b = np.zeros((len(Y), CLASS_NUM))

b[np.arange(len(Y)), Y] = 1

Y =b
# First split the data in two sets, 80% for training, 30% for Val/Test)

X_train, X_valtest, y_train, y_valtest = train_test_split(X,Y, test_size=0.3, random_state=1, stratify=Y)



# Second split the 20% into validation and test sets

X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1, stratify=y_valtest)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

rotation_range=40,

width_shift_range=0.2,

height_shift_range=0.2,

zoom_range=0.2,

horizontal_flip=True,

fill_mode='nearest')



datagen.fit(X_train)
os.makedirs('/tmp/.keras/datasets')
shutil.copytree("../input/keras-pretrained-models", "/tmp/.keras/models")
nb_train_samples = len(X_train)

nb_validation_samples = len(X_val)

epochs = 100

batch_size = 32

print(len(X_train))
#Using VGG16

model = models.Sequential()

model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(CLASS_NUM, activation='softmax'))



vgg16 = applications.VGG16(include_top=False, input_shape=(img_width, img_height, channels), weights='imagenet')



combinedModel = models.Model(inputs= vgg16.input, outputs= model(vgg16.output))



for layer in combinedModel.layers[:-3]:

    layer.trainable = False



combinedModel.compile(loss='categorical_crossentropy',

    optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4),

    metrics=['acc'])



combinedModel.summary()
history = combinedModel.fit_generator(

    datagen.flow(X_train, y_train, batch_size=batch_size),

    steps_per_epoch=len(X_train)//batch_size,

    epochs=epochs,

    validation_data=(X_val, y_val)

)
# summarize history for accuracy

plt.figure(figsize=(15,10))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

# summarize history for loss

plt.figure(figsize=(15,10))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()

score = combinedModel.evaluate(X_test, y_test)

print("Test score:", score[0])

print('Test accuracy:', score[1])
Y_test_class = [list(x).index(1) for x in y_test]

Y_test_class
y_test_class = np.array([list(x).index(1) for x in y_test])

prediction = combinedModel.predict(X_test)
predicted_class_indices=np.argmax(prediction,axis=1)

predicted_class_indices.shape
np.sum(y_test_class==predicted_class_indices)