# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test.head()
X = train.iloc[:,1:]
y = train.iloc[:,0]
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=10, stratify=y)
# Reshape data
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)

test = test.values.reshape(test.shape[0], 28, 28, 1)
# Normalization
X_train = X_train / 255
X_test = X_test / 255

test = test / 255
# One Hot Encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# CNN Network
model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

test_pred = model.predict_classes(test)

sub['Label'] = test_pred

sub.to_csv("submission_v0.csv", index=False)

sub.head()
