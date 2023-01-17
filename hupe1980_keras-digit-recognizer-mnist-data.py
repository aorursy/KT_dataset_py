# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
def load_data():
    df_train  = pd.read_csv("../input/train.csv")
    df_test = pd.read_csv("../input/test.csv")

    y_train = df_train['label'].values
    X_train = df_train.drop(columns=['label']).values
    
    X_test = df_test.values
    
    return (X_train, y_train), (X_test)

(X_train, y_train), (X_test) = load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
input_shape = X_train.shape[1:]

X_train = X_train / 250
X_test = X_test / 250
y_train = to_categorical(y_train)

num_classes = y_train.shape[1] # number of categories
def convolutional_model(num_classes):
    
    # create model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model
# build the model
model = convolutional_model(num_classes)

# fit the model
model.fit(X_train, y_train, validation_split=0.1, epochs=12, batch_size=128, verbose=1)
prediction = model.predict(X_test)
label = np.argmax(prediction, axis=1)
test_id = np.reshape(range(1, len(prediction) + 1), label.shape)
submission = pd.DataFrame({'ImageId': test_id, 'Label': label})
submission.to_csv('submission.csv', index=False)
