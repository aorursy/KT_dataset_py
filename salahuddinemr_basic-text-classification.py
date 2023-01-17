# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







!pip install tensorflow==2.0.0-rc1

import tensorflow as tf

from tensorflow import keras







print(tf.__version__)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
imdb = keras.datasets.imdb 



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000) 
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1]) 
word_index = imdb.get_word_index()
word_index
# The first indices are reserved

word_index = {k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0         # PAD words  int = 0   

word_index["<START>"] = 1       # the start of text  =  int =1 

word_index["<UNK>"] = 2         # unknown words  int =2 

word_index["<UNUSED>"] = 3



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])



def decode_review(text):

    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0])
train_data = keras.preprocessing.sequence.pad_sequences(train_data,

                                                        value=word_index["<PAD>"],

                                                        padding='post',

                                                        maxlen=256)



test_data = keras.preprocessing.sequence.pad_sequences(test_data,

                                                       value=word_index["<PAD>"],

                                                       padding='post',

                                                       maxlen=256)
len(train_data[0]), len(train_data[1])
print(train_data[0])
# input shape is the vocabulary count used for the movie reviews (10,000 words)

vocab_size = 10000



model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, 16))

model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation=tf.nn.relu))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc','binary_crossentropy'])
x_val = train_data[:10000]

partial_x_train = train_data[10000:]



y_val = train_labels[:10000]

partial_y_train = train_labels[10000:]


history = model.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,

                    validation_data=(x_val, y_val),

                    verbose=2)
results = model.evaluate(test_data, test_labels)



print(results)
history_dict = history.history

history_dict.keys()
import matplotlib.pyplot as plt



acc = history_dict['acc']

val_acc = history_dict['val_acc']

loss = history_dict['loss']

val_loss = history_dict['val_loss']



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



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
model_Estop = keras.Sequential()

model_Estop.add(keras.layers.Embedding(vocab_size, 16))

model_Estop.add(keras.layers.GlobalAveragePooling1D())

model_Estop.add(keras.layers.Dense(16, activation=tf.nn.relu))

model_Estop.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))



model_Estop.summary()
model_Estop.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc','binary_crossentropy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) ## activate early stopping

history_Estop = model_Estop.fit(partial_x_train,

                    partial_y_train,

                    epochs=40,

                    batch_size=512,callbacks=[callback],                   ## activate the call back                  

                    validation_data=(x_val, y_val),

                    verbose=2)
results_Estop = model_Estop.evaluate(test_data, test_labels)

print(results_Estop)
history_Estop = history_Estop.history

history_Estop.keys()
plt.clf()   # clear figure

acc = history_Estop['acc']

val_acc = history_Estop['val_acc']

loss = history_Estop['loss']

val_loss = history_Estop['val_loss']



epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()