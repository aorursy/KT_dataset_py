from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Embedding, MaxPooling1D

from tensorflow.keras.preprocessing import sequence

import numpy as np

from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()

max_word = 500

x_train = sequence.pad_sequences(x_train, maxlen=max_word)

x_test = sequence.pad_sequences(x_test, maxlen=max_word)

dict_size = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1
model = Sequential()

model.add(Embedding(dict_size, 128, input_length=max_word))

model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.1))

model.add(Conv1D(filters=256, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.3))

model.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(600, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(300, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

%%time

model.fit(x_train, y_train, 

          validation_data=(x_test, y_test), 

          epochs=10, batch_size=100)
score = model.evaluate(x_test, y_test)

print(score)