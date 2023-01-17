# IMPORT MODULES

# TURN ON the GPU !!!

# If importing dataset from outside - like this IMDB - Internet must be "connected"



import os

from operator import itemgetter    

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

get_ipython().magic(u'matplotlib inline')

plt.style.use('ggplot')



import tensorflow as tf



from keras import models, regularizers, layers, optimizers, losses, metrics

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils, to_categorical

 

from keras.datasets import reuters



print(os.getcwd())

print("Modules imported \n")

print("Files in current directory:")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory

#(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)



import numpy as np

# save np.load

np_load_old = np.load



# modify the default parameters of np.load

np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



# call load_data with allow_pickle implicitly set to true

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)



# restore np.load for future normal usage

np.load = np_load_old
print("train_data ", train_data.shape)

print("train_labels ", train_labels.shape)



print("test_data ", test_data.shape)

print("test_labels ", test_labels.shape)
# Reverse dictionary to see words instead of integers

# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”



word_index = reuters.get_word_index()

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in

train_data[0]])



print(decoded_newswire)

print(train_labels[0])
# VECTORIZE function



def vectorize_sequences(sequences, dimension=10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        results[i, sequence] = 1.

    return results
# Vectorize and Normalize train and test to tensors with 10k columns



x_train = vectorize_sequences(train_data)

x_test = vectorize_sequences(test_data)



print("x_train ", x_train.shape)

print("x_test ", x_test.shape)
# ONE HOT ENCODER of the labels



one_hot_train_labels = to_categorical(train_labels)

one_hot_test_labels = to_categorical(test_labels)



print("one_hot_train_labels ", one_hot_train_labels.shape)

print("one_hot_test_labels ", one_hot_test_labels.shape)
# Setting aside a VALIDATION set



x_val = x_train[:1000]

partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]

partial_y_train = one_hot_train_labels[1000:]



print("x_val ", x_val.shape)

print("y_val ", y_val.shape)



print("partial_x_train ", partial_x_train.shape)

print("partial_y_train ", partial_y_train.shape)
# MODEL



model = models.Sequential()

model.add(layers.Dense(256, kernel_regularizer=regularizers.l1(0.001), activation='relu', input_shape=(10000,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, kernel_regularizer=regularizers.l1(0.001), activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(46, activation='softmax'))



# REGULARIZERS L1 L2

#regularizers.l1(0.001)

#regularizers.l2(0.001)

#regularizers.l1_l2(l1=0.001, l2=0.001)



# Best results I got with HU=128/128/128 or 256/256 and L1=0.001 and Dropout=0.5 = 77.02%

# Without Regularizer 72.92%

# Reg L1 = 76.04, L2 = 76.2, L1_L2 = 76.0

# Only DropOut (0.5) = 76.85%
# FIT / TRAIN model



NumEpochs = 50

BatchSize = 512



model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, epochs=NumEpochs, batch_size=BatchSize, validation_data=(x_val, y_val))



results = model.evaluate(x_val, y_val)

print("_"*100)

print("Test Loss and Accuracy")

print("results ", results)



history_dict = history.history

history_dict.keys()
# VALIDATION LOSS curves



plt.clf()

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
## VALIDATION ACCURACY curves



plt.clf()

acc = history.history['acc']

val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# Retrain from scratch for # of epochs per LEARNING curves above - and evaluate with TEST (which was set aside above)



model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(46, activation='softmax'))



model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(partial_x_train,partial_y_train,epochs= 20, batch_size=512,

validation_data=(x_val, y_val))



results = model.evaluate(x_test, one_hot_test_labels)



print("_"*100)

print(results)
# PREDICT



predictions = model.predict(x_test)

# Each entry in predictions is a vector of length 46

print(predictions[123].shape)



# The coefficients in this vector sum to 1:

print(np.sum(predictions[123]))



# The largest entry is the predicted class — the class with the highest probability:

print(np.argmax(predictions[123]))