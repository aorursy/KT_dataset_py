# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split, StratifiedKFold

import keras

from keras.models import Sequential, Model

from keras.layers import Input, Flatten, Dense, Dropout, Convolution2D, Conv2D, MaxPooling2D, Lambda, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, AveragePooling2D, Concatenate

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.utils import np_utils



%matplotlib inline

keras.backend.set_image_data_format('channels_last')
def load_data_kfold(k):

    

    train = pd.read_json('../input/train.json')

    train.inc_angle = train.inc_angle.replace('na', 0)

    

    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])

    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])

    x_band3 = x_band1 / x_band2

       

    X_train = np.concatenate([x_band1[:, :, :, np.newaxis]

                            , x_band2[:, :, :, np.newaxis]

                            , x_band3[:, :, :, np.newaxis]], axis=-1)

                         

    y_train = np.array(train["is_iceberg"])

    

    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, y_train))

    

    return folds, X_train, y_train



k = 7

folds, X_train, y_train = load_data_kfold(k)
def get_model():

    

    x = Input((75, 75, 3))

    model = BatchNormalization(axis = 3)(x)

    model = Convolution2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model)

    model = MaxPooling2D()(model)

    

    model = BatchNormalization(axis = 3)(model)

    model = Convolution2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model)

    model = MaxPooling2D()(model)

    

    model = BatchNormalization(axis = 3)(model)

    model = Convolution2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model)

    model = MaxPooling2D()(model)

    

    model = BatchNormalization(axis = 3)(model)

    model = Convolution2D(filters = 64, kernel_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(model) 

    model = GlobalAveragePooling2D()(model)

 

    model = Dense(1, activation = 'sigmoid')(model)

    

    model = Model(input = x, output = model)

    

    opt_adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(opt_adam, loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
model = get_model()

model.summary()
batch_size=64



gen = ImageDataGenerator(horizontal_flip = True,

                         vertical_flip = True,

                         width_shift_range = 0.1,

                         height_shift_range = 0.1,

                         zoom_range = 0.1,

                         rotation_range = 10

                        )
def get_callbacks(name_weights, patience_lr):

    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')

    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')

    return [mcp_save, reduce_lr_loss]
for j, (train_idx, val_idx) in enumerate(folds):

    

    print('\nFold ',j)

    X_train_cv = X_train[train_idx]

    y_train_cv = y_train[train_idx]

    X_valid_cv = X_train[val_idx]

    y_valid_cv= y_train[val_idx]

    

    name_weights = "final_model_fold" + str(j) + "_weights.h5"

    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

    generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)

    model = get_model()

    model.fit_generator(

                generator,

                steps_per_epoch=len(X_train_cv)/batch_size,

                epochs=15,

                shuffle=True,

                verbose=1,

                validation_data = (X_valid_cv, y_valid_cv),

                callbacks = callbacks)

    

    print(model.evaluate(X_valid_cv, y_valid_cv))