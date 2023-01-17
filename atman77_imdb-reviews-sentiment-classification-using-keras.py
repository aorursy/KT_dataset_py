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
#Load data and preprocess

from keras.datasets import imdb

from keras.preprocessing import sequence

import numpy as np



imdb_path = r'/kaggle/input/imdb-dataset-for-keras-imdbnpz/imdb.npz'



max_features = 10000

max_len = 500



#Load data

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path=imdb_path, num_words=max_features)



#One hot code

def vectorize_samples(samples, features=max_features):

    results = np.zeros((len(samples), features))

    for i, sample in enumerate(samples):

        results[i, sample] = 1.

    return results

one_hot_train_data = vectorize_samples(train_data)

one_hot_test_data = vectorize_samples(test_data)

one_hot_train_labels = np.asarray(train_labels).astype('float32')

one_hot_test_labels = np.asarray(test_labels).astype('float32')



#Pad sequence

x_train = sequence.pad_sequences(train_data, maxlen=max_len)

x_test = sequence.pad_sequences(test_data, maxlen=max_len)

y_train = np.array(train_labels)

y_test = np.array(test_labels)



print('Data prepared.')
#Common function

import matplotlib.pyplot as plt



def show_loss_and_acc(history):

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

    plt.show()

    plt.figure()

    plt.plot(epochs, acc, 'ro', label='Training acc')

    plt.plot(epochs, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation acc')

    plt.legend()

    plt.show()



def test_model(model, data, labels):

    print('Test result:')

    results = model.evaluate(data, labels)

    predictions = model.predict(data)

    print(results)

    print(predictions)

    

print('Common function prepared.')
#Use functional programming

import keras

from keras import models

from keras import layers

from keras import Input

from keras.models import Model

from keras.utils import plot_model



#Define the model

imdb_input = Input(shape=(None,), dtype='int32', name='imdb')

imdb_embedded = layers.Embedding(max_features, 64)(imdb_input)

branch_a = layers.GRU(128)(imdb_embedded)

branch_b = layers.Conv1D(128, 5, activation='relu')(imdb_embedded)

branch_b = layers.GlobalMaxPooling1D()(branch_b)

concatenated = layers.concatenate([branch_a, branch_b], axis=-1)

imdb_output = layers.Dense(1, activation='sigmoid')(concatenated)

model = Model(imdb_input, imdb_output)

model.summary()



#Compile the model

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])



#monitor the model

callbacks_list = [

    keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)

]

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=callbacks_list)



#Show and test

show_loss_and_acc(history)

test_model(model, x_test, y_test)