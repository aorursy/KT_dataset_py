# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# train data preporcessing

from keras.utils import to_categorical

# train data preporcessing

train_csv = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

x_train = np.array(train_csv.iloc[:, 1:])

y_train = to_categorical(train_csv.iloc[:, 0])

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_train = x_train.astype(np.float32)

x_train /= 255.0

y_train = y_train.astype(np.float32)

print(x_train.shape)

print(x_train.dtype)

print(y_train.shape)

print(y_train.dtype)

# validation data preporcessing

val_csv = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

x_val = np.array(val_csv.iloc[:, 1:])

y_val = to_categorical(val_csv.iloc[:, 0])

x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

x_val = x_val.astype(np.float32)

x_val /= 255.0

y_val = y_val.astype(np.float32)

print(x_val.shape)

print(x_val.dtype)

print(y_val.shape)

print(y_val.dtype)
# # using keras for learning

# import tensorflow as tf

# import keras

# from keras.models import Model

# from keras.layers import Input, Conv2D, Dropout, Activation, Flatten, Dense, MaxPool2D

# from keras.layers.normalization import BatchNormalization

# from keras.optimizers import Adam, SGD, Nadam

# from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, CSVLogger

# from keras.preprocessing.image import ImageDataGenerator



# from adabound import AdaBound



# from random_eraser import get_random_eraser



# # model structure

# inputs = Input(shape=(28, 28, 1), name='input')



# # old model

# conv1_1 = Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv1_1')(inputs)

# bn1 = BatchNormalization(name='bn1')(conv1_1)

# act1 = Activation('relu', name='act1')(bn1)

# conv1_2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv1_2')(act1)

# bn2 = BatchNormalization(name='bn2')(conv1_2)

# act2 = Activation('relu', name='act2')(bn2)

# maxpool1 = MaxPool2D(name='maxpool1')(act2)



# conv2_1 = Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv2_1')(maxpool1)

# bn3 = BatchNormalization(name='bn3')(conv2_1)

# act3 = Activation('relu', name='act3')(bn3)

# conv2_2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv2_2')(act3)

# bn4 = BatchNormalization(name='bn4')(conv2_2)

# act4 = Activation('relu', name='act4')(bn4)

# maxpool2 = MaxPool2D(name='maxpool2')(act4)



# conv3_1 = Conv2D(filters=32, kernel_size=(3,3), padding='same', name='conv3_1')(maxpool2)

# # conv3_2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv3_2')(conv3_1)

# bn5 = BatchNormalization(name='bn5')(conv3_1)

# act5 = Activation('relu', name='act5')(bn5)

# maxpool3 = MaxPool2D(name='maxpool3')(act5)



# # conv4_1 = Conv2D(filters=32, kernel_size=(3,3), padding='same', name='conv4_1')(maxpool3)

# # # conv4_2 = Conv2D(filters=64, kernel_size=(3,3), padding='same', name='conv4_2')(conv4_1)

# # bn6 = BatchNormalization(name='bn6')(conv4_1)

# # act6 = Activation('relu', name='act6')(bn6)

# # maxpool4 = MaxPool2D(name='maxpool4')(act6)



# dropout = Dropout(0.25)(maxpool3)

# flatten = Flatten(name='flatten')(dropout)

# dense1 = Dense(256, activation='relu', name='dense1')(flatten)

# dropout2 = Dropout(0.5)(dense1)

# dense2 = Dense(10, activation='softmax', name='output')(dropout2)



# model = Model(inputs=inputs, outputs=dense2)

# model.summary()

# # local values

# epochs = 1000

# batch_size = 32



# # callbacks

# callbacks = []



# fpath = './checkpoints/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'

# cp_cb = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor='val_acc', verbose=1, save_best_only=False, mode='auto')

# callbacks.append(cp_cb)



# # rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto')

# # callbacks.append(rlp)



# from sgdr import LearningRateCallback

# lr_cbs = LearningRateCallback(1e-3, 1e-4, lr_max_compression=5, t0=10, tmult=2, trigger_val_acc=0.85)

# sgdr = LearningRateScheduler(lr_cbs.lr_scheduler)

# callbacks.append(sgdr)



# import os, datetime

# now = datetime.datetime.now()

# csvname = './log/train_{0:%Y%m%d}_adabound_sgdr_deeper.csv'.format(now)

# csv_logger = CSVLogger(csvname)

# callbacks.append(csv_logger)



# # datagen

# datagen = ImageDataGenerator(width_shift_range=0.2,

#                              height_shift_range=0.2,

#                              rotation_range=5,

#                              zoom_range=[1,1.5],

#                              preprocessing_function=get_random_eraser(v_l=0, v_h=1),

#                              fill_mode="constant", cval=0)

# # datagen.fit(x_train)



# # optimizers

# base_lr = 1e-3

# opt = AdaBound(lr=1e-03,

#                 final_lr=0.1,

#                 gamma=1e-03,

#                 weight_decay=0.,

#                 amsbound=False)

# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size, epochs=epochs, callbacks=callbacks, verbose=1, validation_data=(x_val, y_val))



# model.save('model.h5')
# evaluate

# test_csv = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

# x_test = np.array(test_csv.iloc[:, 1:])

# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# x_test = x_test.astype(np.float32)

# x_test /= 255.0

# id_test = np.array(test_csv.iloc[:, 0])

# from keras.models import load_model



# load your model 



# result_list = []

# for idx in id_test:

#     pred = model.predict(x_test[idx].reshape(1, 28, 28, 1), batch_size=1)

#     result_list.append([idx, np.argmax(pred, axis=1)[0]])

# print(result_list)

# submmit_csv = pd.DataFrame(result_list, index=None, columns=['id', 'label'])

# submmit_csv.to_csv('submission.csv', index=None)