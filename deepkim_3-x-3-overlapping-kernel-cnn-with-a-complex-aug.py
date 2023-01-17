import matplotlib.pyplot as plt
import cv2
import PIL
import tensorflow as tf
import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping

print(os.listdir("../input/fruits-360_dataset/fruits-360"))

base_folder = "../input/fruits-360_dataset/fruits-360"

epochs=4
batch_size = 128
steps_per_epoch_train=37836//128
steps_per_epoch_val=12709//128
datagen_train = ImageDataGenerator(
      rescale=1./255,
      vertical_flip=True,
      horizontal_flip=True,
      shear_range = 0.5,
      zoom_range = 0.2,
      rotation_range=90)
datagen_test = ImageDataGenerator(rescale=1./255)
generator_train = datagen_train.flow_from_directory(directory=base_folder+'/Training',
                               
                                                    target_size=(100,100),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',shuffle=True)

generator_test = datagen_test.flow_from_directory(directory=base_folder+'/Test',
                                                  target_size=(100,100),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)
steps_test = generator_test.n / batch_size
steps_test
from sklearn.utils.class_weight import compute_class_weight
cls_train = generator_train.classes
cls_test = generator_test.classes
from collections import OrderedDict
classes = list(generator_train.class_indices.keys())
num_values = []
unique, counts = np.unique(cls_train, return_counts=True)
valdict=OrderedDict(zip(unique, counts))
for i in range(75):
    num_values.append(valdict[i])
plt.figure(figsize=(30,30))
x = np.arange(len(num_values))
xlabel = classes
plt.bar(x, num_values)
plt.xticks(x, xlabel)
plt.show()    
class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(cls_train), y=cls_train)
class_weight
learning_rate = 2.2279389932900166e-05
num_dense_nodes = 1731
num_epoch = 4
model = Sequential()
  
model.add(Conv2D(64, (3,3), padding='same', input_shape=(100, 100, 3), name='conv2d_1'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
  
model.add(Conv2D(64, (3,3), padding='same', name='conv2d_2'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
model.add(MaxPooling2D(pool_size=2, padding='same', name='maxpool_1'))
  
  
model.add(Conv2D(128, (3,3), padding='same', name='conv2d_3'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
  
model.add(Conv2D(128, (3,3), padding='same', name='conv2d_4'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
model.add(MaxPooling2D(pool_size=2, padding='same', name='maxpool_2'))
  
model.add(Conv2D(256, (3,3), padding='same', name='conv2d_5'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
  
model.add(Conv2D(256, (3,3), padding='same', name='conv2d_6'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
  
model.add(Conv2D(256, (3,3), padding='same', name='conv2d_7'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
model.add(MaxPooling2D(pool_size=2, padding='valid', name='maxpool_3'))
  
model.add(Flatten(name='flatten_1'))
model.add(Dense(num_dense_nodes))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.5))
model.add(Dropout(0.5, name='dropout_1'))
model.add(Dense(75, activation='softmax'))
optimizer=Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model_train = model.fit_generator(generator=generator_train,
                                  epochs=num_epoch,
                                  steps_per_epoch=steps_per_epoch_train,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_per_epoch_val)