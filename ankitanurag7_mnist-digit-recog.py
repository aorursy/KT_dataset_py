# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# os.listdir('../input/digit-recognizer')



# Any results you write to the current directory are saved as output.
from tensorflow import keras



#variables

num_classes = 10

img_rows, img_cols = 28, 28



#obtaining and preprocessing data

input_path = '../input/digit-recognizer/'

dataset = pd.read_csv(input_path + 'train.csv')

#number of rows in dataset

num_images = dataset.shape[0]

X = dataset.values[:,1:].reshape(num_images, img_rows, img_cols, 1) / 255

#use onehotencoding on output

y = keras.utils.to_categorical(dataset.label, num_classes)



#organising test data

test_dataset = pd.read_csv(input_path + 'test.csv')

test_num_images = test_dataset.shape[0]

test_X = test_dataset.values[:, :].reshape(test_num_images, img_rows, img_cols, 1) / 255
#creating the NN model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout



classifier = Sequential()

classifier.add(Conv2D(32, kernel_size = (3, 3), activation='relu',

                      input_shape = (img_rows, img_cols, 1)))

classifier.add(Dropout(0.5))

classifier.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))

classifier.add(Dropout(0.5))

classifier.add(Flatten())

classifier.add(Dense(128, activation='relu'))

classifier.add(Dense(num_classes, activation='softmax'))
#fitting and training model

classifier.compile(optimizer='adam', loss='categorical_crossentropy',

                  metrics=['accuracy'])

classifier.fit(X, y,

              batch_size=128,

              epochs=2,

              validation_split=0.2)
#prediction on test data and formatting

preds = classifier.predict(test_X)

results = np.argmax(preds, axis=1)
#creating output

output = pd.DataFrame({'ImageId': range(1, 28001),

                       'Label': results})

output.to_csv('submission.csv', index=False)