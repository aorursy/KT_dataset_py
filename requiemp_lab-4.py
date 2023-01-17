import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, LSTM
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.datasets import imdb
from keras.preprocessing import sequence
top_words = 5000
max_len = 500
batch_size = 128
epochs = 5
embedding_vector_lenght = 32
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
x_train = sequence.pad_sequences(x_train, maxlen = max_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_len)
model1 = Sequential()
model1.add(Embedding(top_words, embedding_vector_lenght, input_length = max_len))
model1.add(LSTM(100))
model1.add(Dense(1, activation = 'sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])
history1 = model1.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model2 = Sequential()
model2.add(Embedding(top_words, embedding_vector_lenght, input_length = max_len))
model2.add(Conv1D(embedding_vector_lenght, kernel_size=3, padding='same', activation = 'relu'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(LSTM(100))
model2.add(Dense(1, activation = 'sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])
history2 = model2.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model3 = Sequential()
model3.add(Embedding(top_words, embedding_vector_lenght, input_length = max_len))
model3.add(Dropout(0.1))
model3.add(LSTM(100))
model3.add(Dense(1, activation = 'sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])
history3 = model3.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model3.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model4 = Sequential()
model4.add(Embedding(top_words, embedding_vector_lenght, input_length = max_len))
model3.add(Dropout(0.1))
model4.add(Conv1D(embedding_vector_lenght, kernel_size=3, padding='same', activation = 'relu'))
model4.add(MaxPooling1D())
model4.add(Dropout(0.1))
model4.add(LSTM(100))
model4.add(Dense(1, activation = 'sigmoid'))
model4.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics = ['accuracy'])
history4 = model4.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, verbose=2)
score = model4.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])