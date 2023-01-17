# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# Any results you write to the current directory are saved as output.
%pylab inline

import matplotlib.pyplot as plt

import cv2

import numpy as np

from keras.models import Sequential

from keras.layers import Flatten, Dense, Dropout

import os

from keras.utils import normalize

from sklearn.model_selection import train_test_split

from random import shuffle
PATH = '/kaggle/input/four-shapes/shapes/'

classes = ['circle', 'star', 'square', 'triangle']

d = {'circle': 0, 'star': 1, 'square': 2, 'triangle': 3}

x = []

y = []
for c in classes:

    for i in os.listdir(PATH+c):

        img = cv2.imread(PATH + c + '/' + i, 0)

        re_img = cv2.resize(img, (50, 50))

        x.append(re_img)

        y.append(d[c])
# create model

model = Sequential([

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(64, activation='relu'),

    Dropout(0.3),

    Dense(4, activation='sigmoid')

])
# adjust compile

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy']

             )
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
x_train = normalize(x_train, axis=1)

x_test = normalize(x_test, axis=1)
model.fit(x_train, np.asarray(y_train), epochs=5)
model.evaluate(x_test, y_test)
predict = model.predict(x_test)
PREDICT_INDEX = 999
classes[np.argmax(predict[PREDICT_INDEX])]
plt.imshow(x_test[PREDICT_INDEX], cmap='binary')