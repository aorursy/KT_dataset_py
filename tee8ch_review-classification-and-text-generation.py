%pylab

%matplotlib inline

import pandas as pd

import seaborn as sns

import nltk

import string

import os

import re

from __future__ import division, print_function

sns.set_style('white')
from keras.models import Sequential

from keras.layers.noise import GaussianNoise

from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D

from keras.layers.embeddings import Embedding

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical
reviews = pd.read_csv('../input/Reviews.csv')
reviews.head()
reviews.info()
reviews.Score.unique()
reviews.Score.value_counts().plot(kind = 'bar')
%%time



seq_length = 500 #padding/cut to this length

num_samples = 70000

texts = reviews.iloc[:num_samples].Text

#remove html line breaks

text = array([re.sub('<[^<]+?>', '', x) for x in texts])



#labels = reviews.iloc[:50000].Score-1 

labels = reviews.Score.apply(lambda x: x>4).iloc[:num_samples] #shift to start counting with 0



tokenizer = Tokenizer(num_words = 10000)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



data = pad_sequences(sequences, maxlen= seq_length)



labels = labels.astype(int)

#labels = to_categorical(np.asarray(labels), num_classes= 5)

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)



# split the data into a training set and a validation set

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

nb_validation_samples = int(.1 * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]

y_val = labels[-nb_validation_samples:]



print("fraction 5 star in sample: ", sum(labels)/num_samples)

print("fraction 5 star in test set: ", mean(y_val))
data[1]
embeddings_index = {}

f = open(os.path.join('glove.6B/', 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
EMBEDDING_DIM = 100 #given GloVe fileb

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length = seq_length,

                            trainable=True,

                            )
# create the model

embedding_vecor_length = 32

model = Sequential()

#model.add(embedding_layer)

model.add(Embedding(10000, embedding_vecor_length, input_length = data.shape[1]))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(128))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=64)
%%time



text = reviews[reviews.Score == 5].iloc[:5000].Text

# periods as words

#text = array([string.replace(x, '.', " .") for x in text])



#remove html line breaks

text = array([re.sub('<[^<]+?>', '', x) for x in text])



num_words = 800

tokenizer = Tokenizer(num_words = num_words, filters='!"#$%&()*+,.-/:;<=>?@[\\]^_`{|}~\t\n)')

tokenizer.fit_on_texts(text)

concat_text = tokenizer.texts_to_sequences(text)

concat_text = array([item for sublist in concat_text for item in sublist])



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))



# for later text generation: int->word dictionary

inv_word_index = {v: k for k, v in word_index.iteritems()}



## training sequences of length 5

seq_length = 6

dataX = []

dataY = []

for i in range(0, len(concat_text) - seq_length, 1):

    seq_in = concat_text[i:i + seq_length]

    seq_out = concat_text[i + seq_length]

    dataX.append(seq_in)

    dataY.append(seq_out)

n_patterns = len(dataX)

print("Total Patterns: ", n_patterns)
embeddings_index = {}

f = open(os.path.join('glove.6B/', 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
EMBEDDING_DIM = 100 #given GloVe fileb

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights=[embedding_matrix],

                            input_length = seq_length,

                            trainable=True,

                            )
# Small LSTM Network to Generate Text for Alice in Wonderland

# reshape X to be [samples, time steps, features]

#X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

X = numpy.reshape(dataX, (n_patterns, seq_length)) #if embedding is first layer

# normalize

#X = X / float(num_words)

# one hot encode the output variable

y = np_utils.to_categorical(dataY)

# define the LSTM model

embedding_vecor_length = 16

model = Sequential()

model.add(embedding_layer)

# add second LSTM layer

#if True:

    #model.add(Embedding(num_words, embedding_vecor_length, input_length = X.shape[1]))

    #model.add(LSTM(128, return_sequences = True))#, input_shape=(X.shape[1], X.shape[2])))

    #model.add(Dropout(0.2))

model.add(LSTM(256))#, input_shape=(X.shape[1], X.shape[2])))

model.add(Dropout(0.2))

model.add(GaussianNoise(.15))

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

# fit the model

model.fit(X, y, epochs= 4, batch_size = 128, callbacks=callbacks_list)
# load the network weights

filename = "weights-improvement-03-4.2355.hdf5"

model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed

for _ in range(5):

    start = numpy.random.randint(0, len(dataX)-1)

    pattern = dataX[start]

    print("seed:" , ' '.join([inv_word_index[value] for value in pattern]), "...")

    # generate characters

    for i in range(50):

        x = numpy.reshape(pattern, (1, len(pattern)))

        #prediction = model.predict(x, verbose=0)

        #index = numpy.argmax(prediction)

        p = model.predict_proba(x, verbose = 0)[0]

        p[1] /= 1

        top_ind = argsort(p)[::-1][:5] #extract lergest probabilities

        #print top_ind

        p = p[top_ind]

        p /= sum(p)

        #print p

        index = random.choice(top_ind, 1, p = p)[0]

        #print sum(prediction[0]**2)

        result = inv_word_index[index]

        seq_in = [inv_word_index[value] for value in pattern]

        sys.stdout.write(result+' ')

        pattern = append(pattern, index)

        pattern = pattern[1:len(pattern)]

        #print pattern

    print("\n")