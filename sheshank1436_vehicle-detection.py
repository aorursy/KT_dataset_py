# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt  # plotting library
import scipy                     # scientific computnig and technical computing
import cv2                       # working with, mainly resizing, images
import numpy as np               # dealing with arrays
import glob                      # return a possibly-empty list of path names that match pathname
import os                        # dealing with directories
import pandas as pd              # providing data structures and data analysis tools
import tensorflow as tf       
import itertools
import random
from random import shuffle       # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm            # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion
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
train_dir = '../input/vehicle-data-set/cardataset/train'
test_dir = '../input/vehicle-data-set/cardataset/test'
os.listdir(train_dir)
a=os.listdir(train_dir)
for i in a:
    b=os.path.join(train_dir,i)
    print(b)
    c=os.listdir(os.path.join(train_dir,i))
    for j in c:
        d=os.path.join(b,j)
        img=cv2.imread(d)
        print(img.shape)
        
        plt.imshow(img)
        break
LR = 1e-3
height=150
width=150
channels=3
seed=1337
batch_size = 128
num_classes = 17
epochs = 5
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
                                                  shuffle=True,
                                                  class_mode='categorical')

train_num = train_generator.samples
validation_num = validation_generator.samples
x,y=validation_generator.next()
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
os.getcwd()
filepath=str(os.getcwd()+"\cars.h5f")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# = EarlyStopping(monitor='val_acc', patience=15)
callbacks_list = [checkpoint]#, stopper]
history = model.fit_generator(train_generator,
                              steps_per_epoch= train_num // batch_size,
                              epochs=2,
                              validation_data=train_generator,
                              validation_steps= validation_num // batch_size,
                                                              
                             callbacks=callbacks_list,
                              verbose = 1
                             )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()
score = model.evaluate(validation_generator)
score
predictions = model.predict_classes(validation_generator)
predictions[1]
predictions
def predict(out):
    if out==0:
        pred = 'Barge'
    elif out==1:
        pred='Snowmobile'
    elif out==2:
        pred='Tank'
    elif out==3:
        pred='Car'
    elif out==4:
        pred='Truck'
    elif out==5:
        pred='Helicopter'
    elif out==6:
        pred='Bicycle'
    elif out==7:
        pred='Segway'
    elif out==8:
        pred='Cart'
    elif out==9:
        pred='Caterpillar'
    elif out==10:
        pred='Motorcycle'
    elif out==11:
        pred='Ambulance'
    elif out==12:
        pred='Motorcycle'
    elif out==13:
        pred='Ambulance'
    elif out==14:
        pred='Taxi'
    elif out==15:
        pred='Bus'
    elif out==16:
        pred='Van'
    elif out==17:
        pred='Boat'    
    else:
        pred="Limousine"
    return pred
x,y = validation_generator.next()
plt.figure(figsize=(20,10))
for i in range(0,10):
    image = x[i]
    pred=predictions[i]
    actual=y[i]
    plt.subplot(3,3,i+1)
    plt.imshow(image)
    plt.title(f"Predicted: {pred}, \n Class: {actual}")
    plt.show()
