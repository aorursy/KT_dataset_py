from tensorflow.keras.layers import LSTM,Embedding,Dropout,Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.datasets import imdb

import numpy as np


(X_train, y_train), (X_test, y_test) = imdb.load_data()

max_word = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_word)

X_test = sequence.pad_sequences(X_test, maxlen=max_word)

dict_size = np.max([np.max(X_train[i]) for i in range(X_train.shape[0])]) + 1
model = Sequential()

model.add(Embedding(dict_size, 128, input_length=max_word))

model.add(LSTM(128, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()



%%time

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100)
scores = model.evaluate(X_test, y_test)

print(scores)