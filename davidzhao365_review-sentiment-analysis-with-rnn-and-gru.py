# Cache the dataset

from os import listdir, makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

datasets_dir = join(cache_dir, 'datasets')

if not exists(datasets_dir):

    makedirs(datasets_dir)



# If you have multiple input files, change the below cp commands accordingly, typically:

# !cp ../input/keras-imdb-reviews/imdb* ~/.keras/datasets/

!cp ../input/imdb* ~/.keras/datasets/
# Complete Code

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, GRU, Flatten, LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence



# load the dataset but only keep the top n words, zero the rest

top_words = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)



# only take 500 words per review

max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_test = sequence.pad_sequences(X_test, maxlen=max_words)



model = Sequential()

model.add(Embedding(top_words, 100, input_length=max_words))

model.add(GRU(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())



# Train

model.fit(X_train, y_train, epochs=3, batch_size=64)



# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))



# Predict the label for test data

y_predict = model.predict(X_test)
import numpy

from keras.datasets import imdb

from keras.models import Sequential

from keras.layers import Dense, GRU, Flatten, LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence



# load the dataset but only keep the top n words, zero the rest

top_words = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)



max_words = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# word_index is a dictionary mapping words to an integer index

word_index = imdb.get_word_index()

# We reverse it, mapping integer indices to words

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# We decode the review; note that our indices were offset by 3

# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".



def print_review_text(review_one_hot):

    print(review_one_hot)

    review_text = ' '.join([reverse_word_index.get(i - 3, '?') for i in review_one_hot])

    print(review_text)



def print_label_text(label_integer):

    label = "Positive" if label_integer else "Negative"

    print(label)

for review_one_hot, label_integer in zip(X_train[:10], y_train[:10]):

    print_review_text(review_one_hot)

    print("\n")

    print_label_text(label_integer)

    print("\n")
# create the model MLP

model = Sequential()

model.add(Embedding(top_words, 32, input_length=max_words))

model.add(Flatten())

model.add(Dense(250, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
# create the model RNN

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words, embedding_vecor_length, input_length=max_words))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=3, batch_size=64)

# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)

# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))
# Predict the label for test data (pretend this is new reviews)

y_predict = model.predict(X_test)
# Check the predict label and the real label

for review_one_hot, real_label_integer, predict_label_integer in zip(X_test[:10], y_test[:10], y_predict[:10]):

    print_review_text(review_one_hot)

    print("\n")

    print("Actual:")

    print(real_label_integer)

    print("\n")

    print("Predicted:")

    print(predict_label_integer)

    print("\n")