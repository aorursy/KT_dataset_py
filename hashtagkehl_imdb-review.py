# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

from keras import backend as K

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import dataset keras has as part of a package

from keras.datasets import imdb
# prepping the data using only the top 10000 words

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data[0]
train_labels[150]
max([max(sequence)for sequence in train_data])
word_index = imdb.get_word_index() # Dictionary mapping words to integer index

reverse_word_index = dict(

    [(value, key) for (key, value) in word_index.items()])

decoded_review = ' '.join(

    [reverse_word_index.get(i - 3,'?') for i in train_data[0]])

print(decoded_review)
# encoding the integer sequences into a binary matrix

def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.

    return results



x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')

y_test = np.asarray(test_labels).astype('float32')
x_train[0]

# The three layer neural network

from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
# compiling the model using custom losses and metrics. 

from keras import losses

from keras import metrics

from keras import optimizers
# creating a validation set setting apart 10000 samples from the original training set

x_val = x_train[:10000]

partial_x_train = x_train[10000:]

y_val = y_train[:10000]

partial_y_train = y_train[10000:]
# train our model over 20 epochs

# also monitoring the loss and accuracy

model.compile(optimizer=optimizers.RMSprop(lr=0.001),

              loss=losses.binary_crossentropy,

              metrics=[metrics.binary_accuracy])



history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs = 20,

                    batch_size=512,

                    validation_data=(x_val, y_val))

history_dict = history.history

history_dict.keys()

[u'acc', u'loss', u'val_acc', u'val_loss']
# plotting the training and validation loss

import matplotlib.pyplot as plt



history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



epochs = range(1, len(loss_values) + 1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
for key in history.history.keys():

    print(key)

    


plt.clf()

acc = history_dict['binary_accuracy']

val_acc = history_dict['val_binary_accuracy']



plt.plot(epochs, acc, 'bo', label='Training Accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')

plt.title('Training and validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
# retrain the model from scratch

model = models.Sequential()

model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))

model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=0.003),

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
results
Y_=model.predict(x_test)



print(Y_)