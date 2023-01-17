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
import matplotlib.pyplot as plt

import seaborn as sns

import cv2
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head()
train['label'].value_counts()
from sklearn.model_selection import train_test_split
X.shape
from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras.utils import to_categorical
X = train.drop('label', axis=1).values

y = to_categorical(train['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = Sequential()
model.add(Dense(130, activation='relu', input_shape=(784, )))

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=30)
model.evaluate(X_test, y_test)
model.predict_classes(X[3:4])