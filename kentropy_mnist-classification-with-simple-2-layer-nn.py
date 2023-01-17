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

from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(500, input_dim=784, activation='relu'))

model.add(Dense(150, input_dim=784, activation='relu'))

#model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Load training data

data = pd.read_csv('../input/digit-recognizer/train.csv')

y = data.iloc[:, :1]

y_onehot = pd.get_dummies(y['label'])

X = data.iloc[:, 1:]



# Train/test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)

print(X_train.shape)

print(y_train.shape)
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, restore_best_weights=True)

model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test), callbacks=[es])

# Print loss f(x) and value metric (accuracy)

model.evaluate(X_test,y_test)
# Fit with all of the data, and compute the test set predictions.

model.fit(X, y_onehot, epochs=15)

outputs = []

import numpy as np

testdata = pd.read_csv('../input/digit-recognizer/test.csv')

i = 1

for out in model.predict(testdata):

    outputs.append(np.argmax(out))

    i+=1

outs = pd.DataFrame(outputs) 

outs.index = outs.index + 1

print(outs.head)

outs.to_csv("predictions_500_300.csv")