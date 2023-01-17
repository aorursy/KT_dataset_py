!pip install git+https://github.com/qubvel/segmentation_models
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import h5py

import gc

import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras_preprocessing.image import ImageDataGenerator

from keras import models

from keras import layers

from keras import optimizers

from keras import regularizers

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Dropout, GlobalAveragePooling2D

from keras.models import load_model



#model

import efficientnet.keras as efn
train_datagen = ImageDataGenerator(#data_format='channels_first',

                                  validation_split=0.2,

                                  samplewise_center = True,

                                  samplewise_std_normalization = True)



train_generator = train_datagen.flow_from_directory(directory="../input/food116/food/",

                                                    subset="training",

                                                    batch_size=32,

                                                    shuffle=True,

                                                    class_mode="categorical",

                                                    target_size=(299,299),

                                                    seed=42)



valid_generator=train_datagen.flow_from_directory(directory="../input/food116/food/",

                                                  subset="validation",

                                                  batch_size=32,

                                                  shuffle=True,

                                                  class_mode="categorical",

                                                  target_size=(299,299),

                                                  seed=42)
# incnet = efn.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# x = incnet.output

# x = GlobalAveragePooling2D()(x)

# x = Dropout(0.2)(x)

# # x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2())(x)

# # x = Dropout(0.2)(x)

# outputs = Dense(116, activation='softmax', kernel_regularizer=regularizers.l2())(x)



# model = models.Model(incnet.input, outputs)

# model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()
new_model = load_model('../input/b1-model/EfficientNetB1.h5')
gc.collect()
# early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=4)

checkpoint_callback = ModelCheckpoint('EfficientNetB1_2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history = new_model.fit_generator(train_generator,

                            validation_data=valid_generator,

                            epochs=5,workers=0,use_multiprocessing=False, callbacks=[checkpoint_callback])
# new_model.save('mymodel.h5')
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['acc', 'val_acc'])

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['loss', 'val_loss'])

plt.show()