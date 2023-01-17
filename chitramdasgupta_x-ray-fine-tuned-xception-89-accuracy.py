import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('dark')

import sklearn

import tensorflow as tf

from tensorflow import keras
train_dir = '../input/chest-xray-pneumonia/chest_xray/train'

validation_dir = '../input/chest-xray-pneumonia/chest_xray/val'

test_dir = '../input/chest-xray-pneumonia/chest_xray/test'
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')





test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=16,

        class_mode='binary')



test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
from keras.applications import Xception



conv_base = tf.keras.applications.Xception(

    include_top=False,

    weights="imagenet",

    input_shape=(150, 150, 3)

)
conv_base.summary()
conv_base.trainable = True

set_trainable = False



for layer in conv_base.layers:

    if layer.name == 'block14_sepconv1':

        set_trainable = True

        layer.trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False
from keras import optimizers



model = keras.models.Sequential([

    conv_base,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dropout(0.5),

    

    keras.layers.Flatten(),

    

    keras.layers.Dense(128),

    keras.layers.Dropout(0.5),

    keras.layers.BatchNormalization(),

    keras.layers.LeakyReLU(),

    

    keras.layers.Dense(64),

    keras.layers.Dropout(0.5),

    keras.layers.BatchNormalization(),

    keras.layers.LeakyReLU(),

    

    keras.layers.Dense(1, activation='sigmoid')

])



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
model.summary()
my_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)



history = model.fit (

    train_generator,

    steps_per_epoch=5216//20,

    epochs=100,

    validation_data=validation_generator,

    validation_steps=1,

    callbacks=[my_cb]

)
epochs = len(history.history['loss'])

epochs
y1 = history.history['loss']

y2 = history.history['val_loss']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['loss', 'val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.tight_layout()
y1 = history.history['acc']

y2 = history.history['val_acc']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['acc', 'val_acc'])

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.tight_layout()
model.evaluate(validation_generator)
test_loss, test_acc = model.evaluate(test_generator, steps=20)