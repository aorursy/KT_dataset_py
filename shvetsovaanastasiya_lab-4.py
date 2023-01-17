import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM

from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D



from keras.datasets import imdb

from keras.preprocessing import sequence

from keras import metrics



top_words = 5000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)



print(x_train.shape)

print(y_train.shape)
max_world = 500

x_train = sequence.pad_sequences(x_train, maxlen = max_world)

x_test = sequence.pad_sequences(x_test, maxlen = max_world)



num_classes = 2

batch_size = 128

epochs = 3
import matplotlib.pyplot as plt

def plot_val_loss(hist):

    plt.figure(figsize=plt.figaspect(0.5))

    l = range(0, len(hist.history['val_loss']))

    plt.plot(l, hist.history['val_loss'])

    plt.title('val_loss')
embedding_vector_lenght = 32



model_0 = Sequential()

model_0.add(Embedding(top_words, embedding_vector_lenght, input_length = max_world))

model_0.add(LSTM(100))

model_0.add(Dense(1, activation = 'sigmoid'))



model_0.compile(optimizer=keras.optimizers.Adadelta(),

              loss='binary_crossentropy',

#                metrics=[metrics.accuracy, metrics.categorical_accuracy])

                metrics = ['accuracy'])



history_0 = model_0.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model_0.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

model_1 = Sequential()

model_1.add(Embedding(top_words, embedding_vector_lenght, input_length = max_world))



model_1.add(Conv1D(embedding_vector_lenght, 3, activation = 'relu'))

model_1.add(MaxPooling1D())



model_1.add(LSTM(100))

model_1.add(Dense(1, activation = 'sigmoid'))



model_1.compile(optimizer=keras.optimizers.Adadelta(),

              loss='binary_crossentropy',

#                metrics=[metrics.accuracy, metrics.categorical_accuracy])

                metrics = ['accuracy'])



history_1 = model_1.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model_1.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
model_2 = Sequential()

model_2.add(Embedding(top_words, embedding_vector_lenght, input_length = max_world))

model_2.add(Dropout(0.25))

model_2.add(LSTM(100))

model_2.add(Dense(1, activation = 'sigmoid'))



model_2.compile(optimizer=keras.optimizers.Adadelta(),

              loss='binary_crossentropy',

#                metrics=[metrics.accuracy, metrics.categorical_accuracy])

                metrics = ['accuracy'])



history_2 = model_2.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model_2.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
model_3 = Sequential()

model_3.add(Embedding(top_words, embedding_vector_lenght, input_length = max_world))

model_3.add(Conv1D(embedding_vector_lenght, 3, activation = 'relu'))



model_3.add(MaxPooling1D())

model_3.add(Dropout(0.25))

model_3.add(LSTM(100))



model_3.add(Dense(1, activation = 'sigmoid'))



model_3.compile(optimizer=keras.optimizers.Adadelta(),

              loss='binary_crossentropy',

#                metrics=[metrics.accuracy, metrics.categorical_accuracy])

                metrics = ['accuracy'])



history_3 = model_3.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model_3.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])