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
np.random.seed(2016)



import os

import glob

import cv2

import datetime

import time

import warnings

warnings.filterwarnings("ignore")



from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

from keras import __version__ as keras_version
def get_im_cv2(path,im):

    img = cv2.imread(path)

    resized = cv2.resize(img, (im[0],im[1]), cv2.INTER_LINEAR)

    return resized
def load_train(im):

    X_train = []

    X_train_id = []

    y_train = []

    start_time = time.time()



    print('Read train images')

    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    for fld in folders:

        index = folders.index(fld)

        print('Load folder {} (Index: {})'.format(fld, index))

        path = os.path.join('..', 'input', 'train', fld, '*.jpg')

        files = glob.glob(path)

        for fl in files:

            flbase = os.path.basename(fl)

            img = get_im_cv2(fl,im)

            X_train.append(img)

            X_train_id.append(flbase)

            y_train.append(index)



    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return X_train, y_train, X_train_id
def load_test(im):

    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')

    files = sorted(glob.glob(path))



    X_test = []

    X_test_id = []

    for fl in files:

        flbase = os.path.basename(fl)

        img = get_im_cv2(fl,im)

        X_test.append(img)

        X_test_id.append(flbase)



    return X_test, X_test_id

def read_and_normalize_train_data(im):

    train_data, train_target, train_id = load_train(im)



    print('Convert to numpy...')

    train_data = np.array(train_data, dtype=np.uint8)

    train_target = np.array(train_target, dtype=np.uint8)



    print('Reshape...')

    train_data = train_data.transpose((0, 3, 1, 2))



    print('Convert to float...')

    train_data = train_data.astype('float32')

    train_data = train_data / 255

    train_target = np_utils.to_categorical(train_target, 8)



    print('Train shape:', train_data.shape)

    print(train_data.shape[0], 'train samples')

    return train_data, train_target, train_id

def read_and_normalize_test_data(im):

    start_time = time.time()

    test_data, test_id = load_test(im)



    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.transpose((0, 3, 1, 2))



    test_data = test_data.astype('float32')

    test_data = test_data / 255



    print('Test shape:', test_data.shape)

    print(test_data.shape[0], 'test samples')

    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return test_data, test_id
def create_submission(predictions, test_id, info):

    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])

    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)

    now = datetime.datetime.now()

    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    result1.to_csv(sub_file, index=False)
im_1 = [28,28]

im_2 = [56,56]

im_3 = [112,112]

im_4 = [224,224]
train_data_1, train_target, train_id = read_and_normalize_train_data(im_1)

test_data_1, test_id = read_and_normalize_test_data(im_1)
train_data_2, train_target, train_id = read_and_normalize_train_data(im_2)

test_data_2, test_id = read_and_normalize_test_data(im_2)
train_data_3, train_target, train_id = read_and_normalize_train_data(im_3)

test_data_3, test_id = read_and_normalize_test_data(im_3)
train_data_4, train_target, train_id = read_and_normalize_train_data(im_4)

test_data_4, test_id = read_and_normalize_test_data(im_4)
from keras import backend as K

K.image_dim_ordering()



K.set_image_dim_ordering('th')

K.image_dim_ordering()

from keras.layers import Merge

from keras.layers import Dense, Dropout, Activation



model_1 = Sequential()

model_1.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same', input_shape=(3, 28, 28)))

model_1.add(Activation('relu'))

#model_1.add(Convolution2D(32, 3, 3, init='he_normal',border_mode='same'))

#model_1.add(Activation('relu'))

model_1.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_1.add(Dropout(0.25))



model_1.add(Convolution2D(64, 3, 3, init='he_normal',border_mode='same'))

model_1.add(Activation('relu'))

#model_1.add(Convolution2D(64, 3, 3, init='he_normal',border_mode='same'))

#model_1.add(Activation('relu'))

model_1.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_1.add(Dropout(0.25))



#output 64x7x7
model_2 = Sequential()

model_2.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same', input_shape=(3, 56, 56)))

model_2.add(Activation('relu'))

#model_2.add(Convolution2D(32, 5, 5, init='he_normal',border_mode='same'))

#model_2.add(Activation('relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_2.add(Dropout(0.25))



model_2.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))

model_2.add(Activation('relu'))

#model_2.add(Convolution2D(32, 3, 3, init='he_normal',border_mode='same'))

#model_2.add(Activation('relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_2.add(Dropout(0.25))



model_2.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='same'))

model_2.add(Activation('relu'))

#model_2.add(Convolution2D(64, 3, 3, init='he_normal',border_mode='same'))

#model_2.add(Activation('relu'))

model_2.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_2.add(Dropout(0.25))



#output 64x7x7
model_3 = Sequential()

model_3.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same', input_shape=(3, 112, 112)))

model_3.add(Activation('relu'))

#model_3.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same'))

#model_3.add(Activation('relu'))

model_3.add(MaxPooling2D(pool_size=(4, 4),dim_ordering="th"))

model_3.add(Dropout(0.25))



model_3.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))

model_3.add(Activation('relu'))

#model_3.add(Convolution2D(32, 3, 3, init='he_normal', border_mode='same'))

#model_3.add(Activation('relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_3.add(Dropout(0.25))



model_3.add(Convolution2D(64, 3, 3, init='he_normal',  border_mode='same'))

model_3.add(Activation('relu'))

#model_3.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='same'))

#model_3.add(Activation('relu'))

model_3.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_3.add(Dropout(0.25))



#output 64x7x7
model_4 = Sequential()

model_4.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same', input_shape=(3, 224, 224)))

model_4.add(Activation('relu'))

#model_4.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same'))

#model_4.add(Activation('relu'))

model_4.add(MaxPooling2D(pool_size=(4, 4),dim_ordering="th"))

model_4.add(Dropout(0.25))



model_4.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same'))

model_4.add(Activation('relu'))

#model_4.add(Convolution2D(32, 5, 5, init='he_normal', border_mode='same'))

#model_4.add(Activation('relu'))

model_4.add(MaxPooling2D(pool_size=(4, 4),dim_ordering="th"))

model_4.add(Dropout(0.25))



model_4.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='same'))

model_4.add(Activation('relu'))

#model_4.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='same'))

#model_4.add(Activation('relu'))

model_4.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model_4.add(Dropout(0.25))



#output 64x7x7
merged = Merge([model_1, model_2, model_3], mode='concat')



final_model = Sequential()

final_model.add(merged)



final_model.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='valid'))

final_model.add(Activation('relu'))

final_model.add(Dropout(0.25))



final_model.add(Flatten())

final_model.add(Dense(256,init='he_normal'))

final_model.add(Activation('relu'))

final_model.add(Dropout(0.5))

final_model.add(Dense(8, init='he_normal', activation='softmax'))

from keras.optimizers import rmsprop,adam



#rmsprop = rmsprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0001)

final_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])

print(final_model.summary())

history = final_model.fit([train_data_1, train_data_2, train_data_3,], train_target, validation_split=0.33,shuffle=True, nb_epoch=30, batch_size=64, verbose=1)  # we pass one data array per model input

predictions = final_model.predict([test_data_1, test_data_2, test_data_3], batch_size=32, verbose=1)

#model_1.add(Convolution2D(64, 3, 3, init='he_normal', border_mode='valid'))

#model_1.add(Activation('relu'))

#model_1.add(Dropout(0.25))



#model_1.add(Flatten())

#model_1.add(Dense(256,init='he_normal'))

#model_1.add(Activation('relu'))

#model_1.add(Dropout(0.5))

#model_1.add(Dense(8, init='he_normal', activation='softmax'))

#model_1.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics = ['accuracy'])

#print(model_1.summary())

#history = model_1.fit(train_data_1, train_target, validation_split=0.33,shuffle=True, nb_epoch=10, batch_size=32, verbose=1)  # we pass one data array per model input

#predictions = model_1.predict(test_data_1, batch_size=32, verbose=1)





# list all data in history

print(history.history.keys())



# summarize history for accuracy

import matplotlib.pyplot as plt



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
create_submission(predictions, test_id, 'Trail_1')
