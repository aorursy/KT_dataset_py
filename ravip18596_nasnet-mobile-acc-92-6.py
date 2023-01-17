# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense
from keras.models import Sequential
from keras.applications.nasnet import NASNetMobile
from keras.callbacks import EarlyStopping,ModelCheckpoint
from pathlib import Path
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_dir = Path('../input/10-monkey-species/training/training')
test_dir = Path('../input/10-monkey-species/validation/validation')
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv('../input/10-monkey-species/monkey_labels.txt',names=cols,skiprows=1)
labels
labels = labels['Common Name']
labels
import matplotlib.pylab as plt
img = cv2.imread('../input/10-monkey-species/training/training/n0/n0042.jpg')
print(img.shape)
plt.imshow(img)
image_size = 150
channels=3
batch_size= 32

from keras.preprocessing.image import ImageDataGenerator
#train generator
train_generator = ImageDataGenerator(rescale = 1./255)
test_generator = ImageDataGenerator(rescale = 1./255)

training_set = train_generator.flow_from_directory(
                    train_dir,
                    batch_size=batch_size,
                    target_size=(image_size,image_size),
                    class_mode = 'categorical'
                )
test_set = test_generator.flow_from_directory(
                    test_dir,
                    target_size = (image_size,image_size),
                    batch_size=batch_size,
                    class_mode = 'categorical'
                )
keras_models_dir = "../input/keras-models/NASNet-mobile-no-top.h5"
model = Sequential()
model.add(NASNetMobile(include_top=False,weights=keras_models_dir,pooling='avg',input_shape=(image_size,image_size,3)))
model.add(Dense(10,activation='softmax'))
#Compile the model
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics = ["accuracy"]
             )
model.summary()
from keras.callbacks import ReduceLROnPlateau

callbacks = [EarlyStopping(monitor='val_acc',patience=15,verbose=1),
             ModelCheckpoint("nasnet-mobile.h5",monitor='val_acc',save_best_only=True,verbose=1),
             ReduceLROnPlateau(monitor='loss',factor=0.1,patience=2,cooldown=2,min_lr=0.00001,verbose=1)]

hist = model.fit_generator(training_set,
                    steps_per_epoch = 1097 // batch_size,
                    epochs = 30,
                    validation_data = test_set,
                    validation_steps = 272 // batch_size,
                    callbacks = callbacks
                    )
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
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
model.evaluate_generator(test_set,272)
