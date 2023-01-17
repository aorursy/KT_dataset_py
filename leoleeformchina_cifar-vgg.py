# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from __future__ import print_function

import tensorflow as tf

import os



batch_size = 32

num_classes = 10

epochs = 100

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10_trained_model.h5'

log_dir = 'logs/'

checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',

    monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)





# The data, split between train and test sets:

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# Convert class vectors to binary class matrices.

y_train = tf.keras.utils.to_categorical(y_train, num_classes)

y_test = tf.keras.utils.to_categorical(y_test, num_classes)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',

                 input_shape=x_train.shape[1:],kernel_regularizer = tf.keras.regularizers.l2(0.00001)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(32, (3, 3),kernel_regularizer = tf.keras.regularizers.l2(0.00001)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer = tf.keras.regularizers.l2(0.00001)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(64, (3, 3),kernel_regularizer = tf.keras.regularizers.l2(0.00001)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

#model.add(tf.keras.layers.Dropout(0.25))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512,kernel_regularizer = tf.keras.regularizers.l2(0.00001)))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.BatchNormalization())

#model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(num_classes,kernel_regularizer = tf.keras.regularizers.l2(0.00001)))

model.add(tf.keras.layers.Activation('softmax'))



# initiate Adam optimizer

opt = tf.keras.optimizers.Adam(lr=1e-2)



# Let's train the model using Adam

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



model.summary()





x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255





print('Using real-time data augmentation.')

# This will do preprocessing and realtime data augmentation:

datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    zca_epsilon=1e-06,  # epsilon for ZCA whitening

    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

    # randomly shift images horizontally (fraction of total width)

    width_shift_range=0.1,

    # randomly shift images vertically (fraction of total height)

    height_shift_range=0.1,

    shear_range=0.,  # set range for random shear

    zoom_range=0.,  # set range for random zoom

    channel_shift_range=0.,  # set range for random channel shifts

    # set mode for filling points outside the input boundaries

    fill_mode='nearest',

    cval=0.,  # value used for fill_mode = "constant"

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False,  # randomly flip images

    # set rescaling factor (applied before any other transformation)

    rescale=None,

    # set function that will be applied on each input

    preprocessing_function=None,

    # image data format, either "channels_first" or "channels_last"

    data_format=None,

    # fraction of images reserved for validation (strictly between 0 and 1)

    validation_split=0.0)



# Compute quantities required for feature-wise normalization

# (std, mean, and principal components if ZCA whitening is applied).

datagen.fit(x_train)



# Fit the model on the batches generated by datagen.flow().

model.fit_generator(datagen.flow(x_train, y_train,

                                  batch_size=batch_size),

                    epochs=epochs,

                    validation_data=(x_test, y_test),

                    workers=1,

                    callbacks=[logging,checkpoint, early_stopping,reduce_lr])





# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)
# Score trained model.

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])