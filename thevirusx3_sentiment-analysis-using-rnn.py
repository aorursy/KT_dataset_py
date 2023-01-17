import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



from tensorflow.keras.datasets import imdb

from tensorflow.keras.preprocessing.sequence import pad_sequences
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = 20000)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)



print(X_train[:2])
X_train = pad_sequences(X_train, maxlen = 100)

X_test = pad_sequences(X_test, maxlen=100)
vocab_size = 20000

embed_size = 128
from tensorflow.keras import Sequential

from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding
model = Sequential()

model.add(Embedding(vocab_size, embed_size, input_shape = (X_train.shape[1],)))

model.add(LSTM(units=60, activation='tanh'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))
history.history
def plot_learningCurve(history, epochs):

  # Plot training & validation accuracy values

  epoch_range = range(1, epochs+1)

  plt.plot(epoch_range, history.history['accuracy'])

  plt.plot(epoch_range, history.history['val_accuracy'])

  plt.title('Model accuracy')

  plt.ylabel('Accuracy')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Val'], loc='upper left')

  plt.show()



  # Plot training & validation loss values

  plt.plot(epoch_range, history.history['loss'])

  plt.plot(epoch_range, history.history['val_loss'])

  plt.title('Model loss')

  plt.ylabel('Loss')

  plt.xlabel('Epoch')

  plt.legend(['Train', 'Val'], loc='upper left')

  plt.show()
plot_learningCurve(history, 5)