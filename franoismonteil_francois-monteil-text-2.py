from keras.preprocessing import sequence

from sklearn.datasets import fetch_20newsgroups

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense, Embedding, Conv1D, MaxPool1D, Dropout

from keras.layers import Flatten

from keras.preprocessing import sequence

from keras.models import Model
newsgroups_train = fetch_20newsgroups(subset='train')

newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data

y_train = newsgroups_train.target



t_X_test = newsgroups_test.data

X_test = newsgroups_test.data

y_test = newsgroups_test.target
X_train[0]
top_words = 20000

tokenizer = Tokenizer(num_words=top_words)

tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
X_train[0]
max_words = 100

X_train = sequence.pad_sequences(X_train, maxlen=max_words, padding='post')

X_test = sequence.pad_sequences(X_test, maxlen=max_words, padding='post')
from keras.utils import to_categorical

y_test = to_categorical(y_test)

y_train = to_categorical(y_train)
# Exemple extraction de couche : new_model = Model(model.input, model.layers[-2].output)
model = Sequential()

model.add(Embedding(20000,32, input_length=100))

model.add(Conv1D(32, 8, activation="relu"))

model.add(MaxPool1D(2))

model.add(Flatten())

model.add(Dense(20,  activation="softmax"))  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=50, batch_size=128, verbose=2)
new_model = Model(model.input, model.layers[-2].output)