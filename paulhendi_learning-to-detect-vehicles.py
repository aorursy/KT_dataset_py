import glob
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


# Used during testing time
import os
import cv2
from sklearn.metrics import log_loss
import pandas as pd
# General variables 
n_train = len(glob.glob("../input/train/train/**/*.png"))
n_val = len(glob.glob("../input/val/val/**/*.png"))
n_test = len(glob.glob("../input/test/test/*.png"))
batch_size = 64
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory("../input/train/train/",
                                                    target_size=(64,64),
                                                    batch_size = batch_size,
                                                    class_mode = 'binary')

validation_generator = val_datagen.flow_from_directory("../input/val/val/",
                                                         target_size=(64,64),
                                                         batch_size = batch_size,
                                                         class_mode = 'binary')
model = models.Sequential()

model.add(layers.Conv2D(6,(3,3), activation='elu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(12,(3,3), activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(24,(3,3), activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3), activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dense(8, activation = 'elu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.5e-3), metrics=['acc'])
model.summary()
train_steps_per_epoch = int(n_train/batch_size)
val_steps_per_epoch = int(n_val/batch_size)


checkpointer = keras.callbacks.ModelCheckpoint(filepath="model1.hdf5",
                                               save_best_only = True,
                                               monitor = "val_acc")

history = model.fit_generator(train_generator,
                              steps_per_epoch = train_steps_per_epoch,
                              epochs = 20,
                              validation_data = validation_generator,
                              validation_steps = val_steps_per_epoch,
                              callbacks = [checkpointer])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'training acc')
plt.plot(epochs, val_acc, 'b', label = 'validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model.load_weights("model1.hdf5")

submission = pd.DataFrame()
submission['id']= [str(i) + ".png" for i in range(1,n_test+1)]
images = np.array([cv2.imread("../input/test/test/" + str(i) + ".png")[:,:,::-1]/255.0 for i in range(1,n_test+1)])
submission['is_car'] = model.predict(images).flatten()
submission.to_csv("sub.csv", index=False)
# Trying with augmentation of images
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory("../input/train/train/",
                                                    target_size=(64,64),
                                                    batch_size = batch_size,
                                                    class_mode = 'binary')

validation_generator = val_datagen.flow_from_directory("../input/val/val/",
                                                         target_size=(64,64),
                                                         batch_size = batch_size,
                                                         class_mode = 'binary')

# Visualizing output of generator
for image in train_generator :
    plt.imshow(np.reshape(image[0][0], (64,64,3)))
    plt.show()
    break
# Almost the same model with dropout
model = models.Sequential()

model.add(layers.Conv2D(6,(3,3), activation='elu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(12,(3,3), activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(24,(3,3), activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3), activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='elu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(8, activation = 'elu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.5e-3), metrics=['acc'])
model.summary()
train_steps_per_epoch = int(n_train/batch_size)
val_steps_per_epoch = int(n_val/batch_size)

checkpointer = keras.callbacks.ModelCheckpoint(filepath="model2.hdf5",
                                               save_best_only = True,
                                               monitor = "val_acc")

history = model.fit_generator(train_generator,
                              steps_per_epoch = train_steps_per_epoch,
                              epochs = 60,
                              validation_data = validation_generator,
                              validation_steps = val_steps_per_epoch,
                              callbacks = [checkpointer])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'training acc')
plt.plot(epochs, val_acc, 'b', label = 'validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
model.load_weights("model2.hdf5")

submission = pd.DataFrame()
submission['id']= [str(i) + ".png" for i in range(1,n_test+1)]
images = np.array([cv2.imread("../input/test/test/" + str(i) + ".png")[:,:,::-1]/255.0 for i in range(1,n_test+1)])
submission['is_car'] = model.predict(images).flatten()
submission.to_csv("sub2.csv", index=False)