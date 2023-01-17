import numpy as np

import tensorflow as tf

import pandas as pd

import os

import pickle



for i in os.listdir("../input"):

    try:

        data = pd.read_pickle('../input/'+i)

    except Exception as e:

        print(i, "Error loading file", repr(e))



x_train = data["x_train"]

y_train = data["y_train"]

x_test = data["x_test"]

y_test = data["y_test"]
from tensorflow import keras

print(keras.__version__)

from keras_preprocessing.sequence import pad_sequences

# Needed to partiotion the data because of time out.

#partition = round(len(x_train)/2)

partition = 300000

pre_processed_train_x = pad_sequences(x_train[:partition], maxlen=1000)

pre_processed_test_x = pad_sequences(x_test[:partition], maxlen=1000)

input_length = pre_processed_train_x.shape[1]





'''

Check shape dimensions

aval=-1

for train_array in pre_processed_train_x:

    for contender in train_array:

        if contender > aval:

            aval=contender

print(aval)

'''

print("max:", 1000, "min:", min(pre_processed_train_x[2]))

print("input_length:", input_length)

input_dim = 1000 - min(pre_processed_train_x[2])

print("input_dim:", input_dim)
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout



#### OVERRIDE ###

batch_size = 128

output_dim = 256

max_features = data["vocab_size"]



model = keras.Sequential()

model.add(Embedding(input_dim=max_features, output_dim=output_dim, input_length=input_length))

model.add(LSTM(batch_size))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer = 'rmsprop',

             loss = 'binary_crossentropy',

             metrics=['accuracy'])



print(model.summary())

# NB small batch sizes take a long time; batch_size = 512

with tf.device('/GPU:0'):

    history = model.fit(pre_processed_train_x, y_train[:partition], epochs=10, batch_size=256)

    

# save the model

model.save("lstm_model.h5")



val_loss, val_acc = model.evaluate(pre_processed_test_x, y_test[:partition])



import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(model.history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()



plt.plot(model.history.history['loss'], 'orange')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.show()



# Plot training & validation loss values

plt.plot(model.history.history['acc'])

plt.plot(model.history.history['loss'], 'orange')



plt.title('Loss & Accuracy')

plt.xlabel('Epoch')

plt.legend(['Accuracy', 'Loss'], loc='upper left')

plt.show()

print("Loss: ", val_loss, "Accuracy:", val_acc)