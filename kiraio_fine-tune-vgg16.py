# [Preprocessing]
# 1. Download original data
# 2. Organize the data as follows. And upload data(Add Data)
# train/
#  　　daisy/ ←(500/769)
#  　　dandelion/ ←(800/1052)
#  　　rose/ ←(500/784)
#  　　sunflower/ ←(500/734)
#  　　tulip/ ←(700/984)
# validation/
#  　　daisy/ ←(269/769)
#  　　dandelion/ ←(2521052)
#  　　rose/ ←(284/784)
#  　　sunflower/ ←(234/734)
#  　　tulip/ ←(284/984)
# 3. Add Data(VGG16: https://www.kaggle.com/keras/vgg16)

# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import numpy as np
import time

# variable
## vgg16 h5 path
vgg16Path = '/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
## image path
train_data_dir = '/kaggle/input/organizeflowersdataset/change_flowers/change_flowers/train'
validation_data_dir = '/kaggle/input/organizeflowersdataset/change_flowers/change_flowers/validation'
## other
img_width, img_height = 299, 299
nb_train_samples = 100
nb_validation_samples = 800
top_epochs = 50
fit_epochs = 50
batch_size = 24
nb_classes = 5
nb_epoch = 10

#　start measurement
start = time.time()

# import vgg16 model
input_tensor = Input(shape=(img_width, img_height, 3))
vgg16 = keras.applications.VGG16(weights=vgg16Path, include_top=False, input_tensor=input_tensor)

# creating an FC layer
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

# bound VGG 16 and FC layer
vgg_model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# prevent re-learning of the layer before the last convolution layer
for layer in vgg_model.layers[:15]:
    layer.trainable = False

# create model
vgg_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy']
)

# Setting learning data
train_datagen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
)

history = vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples
)

process_time = (time.time() - start) / 60
print(u'Learning done. Time spent: ', process_time, u'Min')