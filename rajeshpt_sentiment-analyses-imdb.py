import numpy as np

import pandas as pd

from keras.datasets import imdb

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Input
top_words = 5000

(train_features, train_labels), (test_features, test_labels) = imdb.load_data(num_words = top_words)
max_words = 500

train_features = sequence.pad_sequences(train_features, maxlen = max_words)

test_features = sequence.pad_sequences(test_features, maxlen = max_words)
model = Sequential()

model.add(Embedding(top_words, 32, input_length=max_words))

model.add(Flatten())

model.add(Dense(250, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_features, train_labels, validation_data=(test_features, test_labels), epochs=2, batch_size=128, verbose=2)

scores = model.evaluate(test_features, test_labels, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))