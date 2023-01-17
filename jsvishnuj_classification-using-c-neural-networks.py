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
# importing all the essential libraries for the Neural Networks

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
# importing the training and testing data

train_data = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test_data = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
# creating training and testing datasets with labels

X_train = train_data.drop('label', axis = 1)

y_train = train_data['label']



X_test = test_data.drop('label', axis = 1)

y_test = test_data['label']
X_train.shape
# normalizing the data

X_train = X_train / 255

X_test = X_test / 255
# visualizing any one image for our imagination :) 

plt.imshow(X_train.iloc[:1].values.reshape((28, 28)))
rows, columns = 28, 28

X_train = X_train.values.reshape(X_train.shape[0], rows, columns, 1)

X_test = X_test.values.reshape(X_test.shape[0], rows, columns, 1)
X_train.shape
CNN_model = Sequential()

# convolutional layer - 1

CNN_model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (rows, columns, 1)))

# Max pooling Layer - 1

CNN_model.add(MaxPool2D(pool_size = (2, 2)))

# Drop out layer - 1

CNN_model.add(Dropout(0.2))

# convolutional layer - 2

CNN_model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

# Max pooling Layer - 2

CNN_model.add(MaxPool2D(pool_size = (2, 2)))

# Drop out layer - 2

CNN_model.add(Dropout(0.2))

# convolutional layer - 3

CNN_model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))

# Max pooling Layer - 3

CNN_model.add(MaxPool2D(pool_size = (2, 2)))

# Drop out layer - 3

CNN_model.add(Dropout(0.2))

# flatten layer

CNN_model.add(Flatten())

# fully connected layer 1

CNN_model.add(Dense(units = 128, activation = 'relu'))

# Drop out layer - 4

CNN_model.add(Dropout(0.2))

# output layer

CNN_model.add(Dense(units = len(set(y_train)), activation = 'softmax'))

# compiling the model

CNN_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
epochs = 25

fitted_model = CNN_model.fit(X_train, y_train, epochs = epochs, verbose = 1, validation_split = 0.2)
# summary of the model

CNN_model.summary()
prediction = CNN_model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(prediction, y_test))
print(confusion_matrix(prediction, y_test))
print(accuracy_score(prediction, y_test))
accuracy = fitted_model.history['acc']

validated_accuracy = fitted_model.history['val_acc']



plt.plot(range(len(validated_accuracy)), validated_accuracy, 'go', label = 'validated_accuracy')

plt.plot(range(len(accuracy)), accuracy, 'ro', label = 'accuracy')



plt.title('Training vs validated accuracy')



plt.legend()