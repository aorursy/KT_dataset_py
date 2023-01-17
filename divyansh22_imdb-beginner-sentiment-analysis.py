from keras.datasets import imdb

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense

from keras.layers.embeddings import Embedding

from keras.layers import Flatten

from keras.callbacks import EarlyStopping

import string

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
top_words = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
word_dict = imdb.get_word_index()

word_dict = { key:(value + 3) for key, value in word_dict.items() }

word_dict[''] = 0                                                    # Padding

word_dict['>'] = 1                                                   # Start

word_dict['?'] = 2                                                   # Unknown word

reverse_word_dict = { value:key for key, value in word_dict.items() }

print(' '.join(reverse_word_dict[id] for id in x_train[0]))
max_review_length = 500

x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)

x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
embedding_vector_length = 32

model = Sequential()

model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

model.add(Flatten())

model.add(Dense(16, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
es = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=100)

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=128, callbacks=[es], verbose=0)



_, train_acc = model.evaluate(x_train, y_train, verbose=1)

_, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
%matplotlib inline



sns.set()

acc = hist.history['accuracy']

val = hist.history['val_accuracy']

epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, '-', label='Training accuracy')

plt.plot(epochs, val, ':', label='Validation accuracy')

plt.title('Training and Validation Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(loc='upper left')

plt.plot()
%matplotlib inline



sns.set()

loss = hist.history['loss']

val = hist.history['val_loss']

epochs = range(1, len(loss) + 1)



plt.plot(epochs, loss, '-', label='Training loss')

plt.plot(epochs, val, ':', label='Validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(loc='upper left')

plt.plot()
scores = model.evaluate(x_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1] * 100))
def analyze(text):

    # Prepare the input by removing punctuation characters, converting

    # characters to lower case, and removing words containing numbers

    translator = str.maketrans('', '', string.punctuation)

    text = text.translate(translator)

    text = text.lower().split(' ')

    text = [word for word in text if word.isalpha()]



    # Generate an input tensor

    input = [1]

    for word in text:

        if word in word_dict and word_dict[word] < top_words:

            input.append(word_dict[word])

        else:

            input.append(2)

    padded_input = sequence.pad_sequences([input], maxlen=max_review_length)



    # Invoke the model and return the result

    result = model.predict(np.array([padded_input][0]))[0][0]

    return result
analyze('Easily the most stellar experience I have ever had.')
analyze('This is the shittiest thing I have ever heard!') # Perhaps "shittiest" would be an outlier. (Need to take care of in future models!) 
analyze('I had a really bad experience with the customer service.')
analyze('This film is a once-in-a-lifetime opportunity')