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
import tensorflow as tf



tf.__version__



import math                      

import matplotlib.pyplot as plt  

import scipy                     

import cv2                       

import numpy as np               

import glob                      

import os                        

import pandas as pd              

import tensorflow as tf       

import itertools

import random

from random import shuffle       

from tqdm import tqdm            

from PIL import Image

from scipy import ndimage

from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics

%matplotlib inline

np.random.seed(1)



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras.callbacks import ModelCheckpoint, EarlyStopping



from tensorflow import keras

from tensorflow.keras.callbacks import TensorBoard 



import numpy as np

from time import time

import matplotlib.pyplot as plt

train_dir = Path('../input/training/training/')

test_dir = Path('../input/validation/validation/')

columns = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']

labels = pd.read_csv("../input/monkey_labels.txt", names=columns, skiprows=1)

labels
def image_show(num_image,label):

    for i in range(num_image):

        imgdir = Path('../input/training/training/' + label)

        #print(imgdir)

        imgfile = random.choice(os.listdir(imgdir))

        #print(imgfile)

        img = cv2.imread('../input/training/training/'+ label +'/'+ imgfile)

       # print(img.shape)

        #print(label)

        plt.figure(i)

        plt.imshow(img)

        plt.title(imgfile)

    plt.show()
labels = labels['Common Name']

labels
print(labels[7])

image_show(3,'n7')


height=150

width=150

channels=3

seed=1337

batch_size = 64

num_classes = 10

epochs =150

data_augmentation = True

num_predictions = 20



# Training generator

train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')



train_generator = train_datagen.flow_from_directory(train_dir, 

                                                    target_size=(height,width),

                                                    batch_size=batch_size,

                                                    seed=seed,

                                                    shuffle=True,

                                                    class_mode='categorical')



# Test generator

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(test_dir, 

                                                  target_size=(height,width), 

                                                  batch_size=batch_size,

                                                  seed=seed,

                                                  shuffle=False,

                                                  class_mode='categorical')



train_num = train_generator.samples

validation_num = validation_generator.samples
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc'])

model.summary()
filepath=str(os.getcwd()+"/model.h5f")

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# = EarlyStopping(monitor='val_acc', patience=15)

callbacks_list = [checkpoint]#, stopper]
model_history = model.fit_generator(train_generator,

                              steps_per_epoch= train_num // batch_size,

                              epochs=150,

                              validation_data=train_generator,

                              validation_steps= validation_num // batch_size,

                              callbacks=callbacks_list, 

                              verbose = 1

                             )
acc = model_history.history['acc']

val_acc = model_history.history['val_acc']

loss = model_history.history['loss']

val_loss = model_history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'orange', label='Training acc')

plt.plot(epochs, val_acc, 'green', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'orange', label='Training loss')

plt.plot(epochs, val_loss, 'green', label='Validation loss')



plt.legend()



plt.show()
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy

    if cmap is None:

        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float32') / cm.sum(axis=1)

        cm = np.round(cm,2)

        



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.2f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))

    plt.show()
from keras.models import load_model

model_trained = load_model(filepath)

# Predict the values from the validation dataset

Y_pred = model_trained.predict_generator(validation_generator, validation_num // batch_size+1)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis = 1)

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true = validation_generator.classes,y_pred = Y_pred_classes)

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)

print(metrics.classification_report(validation_generator.classes, Y_pred_classes,target_names=labels))