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
# Define model

import keras

from keras.models import Sequential

from keras.layers import BatchNormalization, Conv1D, Convolution2D, MaxPooling1D, MaxPooling2D, Lambda, Dense, Dropout, Flatten

model = Sequential()

model.add(Convolution2D(32,(3,3), activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization(axis=2))

model.add(Convolution2D(32,(3,3), activation='relu'))

model.add(MaxPooling2D())

model.add(BatchNormalization(axis=2))

model.add(Convolution2D(64,(3,3), activation='relu'))

model.add(BatchNormalization(axis=2))

model.add(Convolution2D(64,(3,3), activation='relu'))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(500, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save_weights('empty_network.h5')
# Load training data

data = pd.read_csv('../input/digit-recognizer/train.csv')

y = data.iloc[:, :1]

y_onehot = pd.get_dummies(y['label'])

X = data.iloc[:, 1:]

X = X.to_numpy()





# Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

#X_train = np.expand_dims(X_train, axis=2)

#X_test = np.expand_dims(X_test, axis=2)

print(X_train.shape)

print(y_train.shape)
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, restore_best_weights=True)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[es])

# Print loss f(x) and value metric (accuracy)

model.evaluate(X_test,y_test)
# Fit with all of the data, and compute the test set predictions.

#model.load_weights('empty_network.h5')

#X = X.reshape(X.shape[0], 28, 28,1)

#model.fit(X, y_onehot, epochs=15)

outputs = []

import numpy as np

testdata = pd.read_csv('../input/digit-recognizer/test.csv')

testdata = testdata.to_numpy()

testdata = testdata.reshape(testdata.shape[0], 28, 28,1)

i = 1

for out in model.predict(testdata):

    outputs.append(np.argmax(out))

    i+=1

outs = pd.DataFrame(outputs) 

outs.index = outs.index + 1

print(outs.head)

outs.to_csv("predictions_cnn.csv")