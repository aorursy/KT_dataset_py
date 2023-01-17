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
from sklearn.model_selection import train_test_split



VAL_SIZE = 0.20

SEED=100



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



X_train, X_val, y_train, y_val = train_test_split(train.drop(['label'],axis=1),train['label'],test_size=VAL_SIZE,stratify=train['label'], random_state=SEED)

print (X_train.shape, X_val.shape, y_train.shape, y_val.shape)



train_dataset = X_train.values

train_labels = y_train



valid_dataset = X_val.values

valid_labels = y_val



train_dataset = train_dataset.reshape(X_train.shape[0], 28, 28, 1)

valid_dataset = valid_dataset.reshape(X_val.shape[0], 28, 28, 1)



train_dataset = train_dataset.astype('float32')

train_dataset/= 255



valid_dataset = valid_dataset.astype('float32')

valid_dataset /= 255
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils



train_labels = np_utils.to_categorical(train_labels, 10)

valid_labels = np_utils.to_categorical(valid_labels, 10)
NUM_EPOCHS = 1

BATCH_SIZE = 128



from keras.optimizers import Adam



def create_model():

    model = Sequential()



    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(28, 28, 1)))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))

    model.add(MaxPooling2D(pool_size=(2,2)))



    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))

    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))

    model.add(MaxPooling2D(pool_size=(2,2)))

     

    model.add(Flatten())

    model.add(Dropout(0.2))

    model.add(Dense(4096, activation='relu'))

    

    model.add(Dense(10, activation='softmax'))



    adam = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

run_model = create_model()

run_model.summary()