from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation
from keras.layers import LSTM, Conv1D, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
batch_size = 128
top_words = 5000
max_review_length = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
# Построить классификатор с использованием свертки

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), batch_size=batch_size)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Train on 25000 samples, validate on 25000 samples
# Epoch 1/2
# 25000/25000 [==============================] - 55s 2ms/step - loss: 0.4700 - accuracy: 0.7617 - val_loss: 0.2883 - val_accuracy: 0.8800
# Epoch 2/2
# 25000/25000 [==============================] - 56s 2ms/step - loss: 0.2410 - accuracy: 0.9054 - val_loss: 0.2981 - val_accuracy: 0.8801
# Accuracy: 88.01%
# Построить вариант классификатора без сверточных слоев

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), batch_size=batch_size)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Train on 25000 samples, validate on 25000 samples
# Epoch 1/2
# 25000/25000 [==============================] - 110s 4ms/step - loss: 0.5299 - accuracy: 0.7273 - val_loss: 0.4184 - val_accuracy: 0.8185
# Epoch 2/2
# 25000/25000 [==============================] - 108s 4ms/step - loss: 0.3400 - accuracy: 0.8570 - val_loss: 0.3348 - val_accuracy: 0.8619
# Accuracy: 86.19%
# Добавить dropout и оценить его влияние

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), batch_size=batch_size)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Train on 25000 samples, validate on 25000 samples
# Epoch 1/2
# 25000/25000 [==============================] - 54s 2ms/step - loss: 0.5796 - accuracy: 0.6807 - val_loss: 0.3670 - val_accuracy: 0.8338
# Epoch 2/2
# 25000/25000 [==============================] - 54s 2ms/step - loss: 0.2784 - accuracy: 0.8872 - val_loss: 0.2743 - val_accuracy: 0.8857
# Accuracy: 88.57%