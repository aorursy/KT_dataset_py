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
from keras.datasets import imdb



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_labels[0]
def vectorize_sequences(sequences, dimension=10000):

    # Create an all-zero matrix of shape (len(sequences), dimension)

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1  #set specific indices of results[i] to 1s

    return results



# Our vectorized training data

x_train = vectorize_sequences(train_data)

# Our vectorized test data

x_test = vectorize_sequences(test_data)

x_train[0]
# Our vectorized labels

y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
y_train
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras import optimizers



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras import losses

from keras import metrics



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss=losses.binary_crossentropy,

              metrics=[metrics.binary_accuracy])
x_val = x_train[:10000]

partial_x_train = x_train[10000:]



y_val = y_train[:10000]

partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=20,

                    batch_size=512,

                    validation_data=(x_val, y_val))
import matplotlib.pyplot as plt



acc = history.history['loss']

val_acc = history.history['binary_accuracy']

loss = history.history['val_loss']

val_loss = history.history['val_binary_accuracy']



epochs = range(1, len(acc) + 1)



# "bo" is for "blue dot"

plt.plot(epochs, loss, 'bo', label='Training loss')

# b is for "solid blue line"

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
plt.clf()   # clear figure

acc_values = history.history['binary_accuracy']

val_acc_values = history.history['val_binary_accuracy']



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
model = models.Sequential()

model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
