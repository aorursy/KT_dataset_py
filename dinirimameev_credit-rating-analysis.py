# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the dataset

dataset = pd.read_excel('../input/credit-history/Credit History.xlsx')

dataset.head()



# Fill gaps

for i in dataset.columns:

    dataset[i] = dataset[i].replace(' ', -1)
# Remove unnecessary columns from DataFrame

labeled_data = dataset.drop(['NaturalPersonID', 'RequestDate'], axis=1)

labeled_data.head()
# Data(X) and labels(Y)

X = labeled_data.drop(['Target'], axis=1)

Y = labeled_data['Target']



# Split to train and test datasets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=54321,

                                                    test_size=0.2)



# Convert to np array

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values



# Normalize output vectors

y_train_bin = np_utils.to_categorical(y_train)

y_test_bin = np_utils.to_categorical(y_test)
# Create model

with tf.device('/GPU:0'):

    model = Sequential()

    model.add(Dense(9, input_dim=215, activation='relu')) 

    model.add(Dense(10, activation='relu', ))

    model.add(Dense(10, activation='relu', ))

    model.add(Dense(10, activation='relu', ))

    model.add(Dense(30, activation='relu', ))

    model.add(Dense(10, activation='relu', ))

    model.add(Dense(10, activation='relu', ))

    model.add(Dense(10, activation='relu', ))

    model.add(Dense(2, activation='softmax')) # 1 output class final probability





    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train a model

with tf.device('/GPU:0'):

    model.fit(X_train, y_train_bin, 

              validation_data=(X_test, y_test_bin),

              shuffle=True,

              epochs=300, 

              batch_size=512)
# evaluate the model

scores = model.evaluate(X_test, y_test_bin)

print("\nAccuracy: %.2f%%" % (scores[1]*100))