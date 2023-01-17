from keras.models import Sequential

from keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Dropout

from keras.layers import Flatten

from keras.preprocessing import sequence
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train[0]
max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')

X_test = sequence.pad_sequences(X_test, maxlen=max_words, padding='post')
X_train[0]
model = Sequential()

model.add(Embedding(10000,32, input_length=500))

model.add(Flatten())

model.add(Dense(250, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=128, verbose=2)
model2 = Sequential()

model2.add(Embedding(10000,32, input_length=500))

model2.add(Dropout(0.2))

model2.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))

model2.add(MaxPool1D(pool_size=3))

model2.add(Dropout(0.2))

model2.add(Conv1D(filters=128, kernel_size=2, padding='same', activation='relu'))

model2.add(MaxPool1D(pool_size=3))

model2.add(Dropout(0.2))

model2.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))

model2.add(MaxPool1D(pool_size=3))

model2.add(Dropout(0.2))

model2.add(Flatten())

model2.add(Dropout(0.2))

model2.add(Dense(250, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
model2.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=128, verbose=2)