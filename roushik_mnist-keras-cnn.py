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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
y = train['label']



X = train.drop('label', axis=1)
X.shape
X = X.values.reshape(-1, 28, 28, 1)
X.shape
# Scaling

X = X / 255
from tensorflow.keras.utils import to_categorical



y = to_categorical(y)
y[0]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=26)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
model = Sequential()



model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPool2D())

model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))

model.add(Dropout(0.2))



model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPool2D())

model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))

model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
test.shape
test = test.values.reshape(-1, 28, 28, 1)
test = test / 255
results = model.predict(test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)