# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import csv

import cv2

import os, glob



import sklearn

from sklearn.model_selection import train_test_split

import time

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten, Activation

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D,ZeroPadding2D

from keras.utils import np_utils

from matplotlib import pyplot as plt

from keras.optimizers import RMSprop, Adam

from keras.callbacks import EarlyStopping

from keras import backend as K
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def load_data(data_dir):

    

    # Get all subdirectories of data_dir. Each represents a label.

    directories = [d for d in os.listdir(data_dir)

                   if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in

    # two lists, labels and images.

    labels = []

    images = []



    category = 0

    for d in directories:

        label_dir = os.path.join(data_dir, d)

        file_names = [os.path.join(label_dir, f)

                      for f in os.listdir(label_dir)

                      if f.endswith(".jpg")]

        

        # adding an early stop for sake of speed

        #stop = 0

        for f in file_names:

            img = cv2.imread(f)

            imresize = cv2.resize(img, (200,125))

            #plt.imshow(imresize)

            images.append(imresize)

            labels.append(category)

            # remove this to use full data set

            '''

            if stop > 30:

                break

            stop += 1'

            # end early stop

            '''

        category += 1

        

    return images, labels



data_dir = "../input/kkkkkkkkkkkkkkkkkkk/train"

images, labels = load_data(data_dir)



# confirm that we have the data

print(images[0:10])

print(labels[0:10])
len(images)
from sklearn.model_selection import train_test_split





def cross_validate(Xs, ys):

    X_train, X_test, y_train, y_test = train_test_split(

            Xs, ys, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test



X_train, X_valid, y_train, y_valid = cross_validate(images, labels)



# confirm we got our data

print(y_valid[0:10])
X_train = np.array(X_train).astype('float32')

X_valid = np.array(X_valid).astype('float32')

X_train = X_train / 255.0

X_valid = X_valid / 255.0
X_train.shape
y_train = np.array(y_train)

y_valid = np.array(y_valid)

y_train = np_utils.to_categorical(y_train)

y_valid = np_utils.to_categorical(y_valid)

num_classes = y_valid.shape[1]
num_classes
optimizer = RMSprop(lr=1e-4)

objective = 'categorical_crossentropy'



def center_normalize(x):

    return (x - K.mean(x)) / K.std(x)

model = Sequential()



model.add(Activation(activation=center_normalize, input_shape=(125,200,3)))

#model.add(Convolution2D(32, 3, 3, input_shape=(250,400, 3), border_mode='same', activation='relu', W_constraint=maxnorm(3)))



model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))



model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))



model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))



model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))

model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))





model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_classes))

model.add(Activation('sigmoid'))



model.compile(loss=objective, optimizer=optimizer)



early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto') 
model.fit(X_train, y_train, batch_size=64, nb_epoch=30,

              validation_split=0.2, verbose=1, shuffle=True, callbacks=[early_stopping])
from sklearn.metrics import log_loss



preds = model.predict(X_valid, verbose=1)

print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))
path1  = '../input/kalashkalash/test_stg1'

file_names1 = [os.path.join(path1, f)

                      for f in os.listdir(path1)

                      if f.endswith(".jpg")]

images1 = []

for f in file_names1:

    img = cv2.imread(f)

    imresize = cv2.resize(img, (200, 125))

    #plt.imshow(imresize)

    images1.append(imresize)
path2  = '../input/testdataset/test_stg2/test_stg2'

file_names2 = [os.path.join(path2, f)

                      for f in os.listdir(path2)

                      if f.endswith(".jpg")]

for f in file_names2:

    img = cv2.imread(f)

    imresize = cv2.resize(img, (200, 125))

    #plt.imshow(imresize)

    images1.append(imresize)
test = np.array(images1).astype('float32')

test = test/255.0

test.shape
test_preds = model.predict(test, verbose=1)

imfile = []

for f in file_names1:

    k = f.split('/')

    imfile.append(k[-1])

    
len(imfile)
for f in file_names2:

    k = f.split('/')

    imfile.append('test_stg2/'+k[-1])
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)

submission.insert(0, 'image', imfile)

submission[1005:1010]
submission.shape
submission.to_csv('mycsvfile2.csv',index=False)