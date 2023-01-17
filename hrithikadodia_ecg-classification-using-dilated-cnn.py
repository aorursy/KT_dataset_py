# Importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import keras

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

from keras.models import Sequential

from sklearn.utils import shuffle

from keras.utils import to_categorical
# Reading Data 

train_data = pd.read_csv('/kaggle/input/heartbeat/mitbih_train.csv', header = None)

test_data = pd.read_csv('/kaggle/input/heartbeat/mitbih_test.csv', header = None)
len(train_data.columns)
train_data[187] = train_data[187].astype('int32')

test_data[187] = test_data[187].astype('int32')
X_train = np.array(train_data.iloc[:, :187])

X_test = np.array(test_data.iloc[:, :187])

y_train = np.array(train_data[187])

y_test = np.array(test_data[187])
X_train, y_train = shuffle(X_train, y_train, random_state = 101)

X_test, y_test = shuffle(X_test, y_test, random_state = 101)
X_train = np.expand_dims(X_train, 2)

X_test = np.expand_dims(X_test, 2)
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
listX = X_train[:5]

plt.figure(figsize = [8, 5])

for x in listX:

    plt.plot(np.arange(0, 187), x)
model = Sequential()

model.add(Conv1D(32, (3), input_shape = (187, 1), activation = 'relu'))

model.add(Conv1D(32, (3), activation = 'relu'))

model.add(MaxPooling1D(2))

model.add(Conv1D(64, (5), dilation_rate = (2), activation = 'relu'))

model.add(Conv1D(64, (5), activation = 'relu'))

model.add(MaxPooling1D(2))

model.add(Conv1D(128, (5), dilation_rate = (2), activation = 'relu'))

model.add(Conv1D(128, (5), activation = 'relu'))

model.add(MaxPooling1D(2))

model.add(Flatten())

model.add(Dense(64, activation = 'relu'))

model.add(Dense(5, activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 5, batch_size = 100, validation_data = (X_test, y_test))
plt.figure(figsize = [7, 5])

plt.plot(history.history['accuracy'], label = 'Training')

plt.plot(history.history['val_accuracy'], label = 'Validation')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Plot Accuracy')
plt.figure(figsize = [7, 5])

plt.plot(history.history['loss'], label = 'Training')

plt.plot(history.history['val_loss'], label = 'Validation')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Plot Loss')