from __future__ import print_function



import os

import sys

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.layers import Dense, Input, GlobalMaxPooling1D

from keras.layers import Conv1D, MaxPooling1D, Embedding

from keras.models import Model

from keras.initializers import Constant
MAX_SEQUENCE_LENGTH = 1000

MAX_NUM_WORDS = 20000

EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2
embeddings_index = {}

with open(os.path.join("/kaggle/input/glove6b/", 'glove.6B.100d.txt')) as f:

    for line in f:

        word, coefs = line.split(maxsplit=1)

        coefs = np.fromstring(coefs, 'f', sep=' ')

        embeddings_index[word] = coefs



print('Found %s word vectors.' % len(embeddings_index))
texts = [] 

labels_index = {}  

labels = [] 
for name in sorted(os.listdir("/kaggle/input/newsdata-set/20_newsgroup/")):

    path = os.path.join("/kaggle/input/newsdata-set/20_newsgroup/", name)

    if os.path.isdir(path):

        label_id = len(labels_index)

        labels_index[name] = label_id

        for fname in sorted(os.listdir(path)):

            if fname.isdigit():

                fpath = os.path.join(path, fname)

                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}

                with open(fpath, **args) as f:

                    t = f.read()

                    i = t.find('\n\n')  # skip header

                    if 0 < i:

                        t = t[i:]

                    texts.append(t)

                labels.append(label_id)



print('Found %s texts.' % len(texts))
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-num_validation_samples]

y_train = labels[:-num_validation_samples]

x_val = data[-num_validation_samples:]

y_val = labels[-num_validation_samples:]
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)

embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():

    if i >= MAX_NUM_WORDS:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
embedding_layer = Embedding(num_words,

                            EMBEDDING_DIM,

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=False)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(5)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = GlobalMaxPooling1D()(x)

x = Dense(128, activation='relu')(x)

preds = Dense(len(labels_index), activation='softmax')(x)
model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])
model.fit(x_train, y_train,

          batch_size=128,

          epochs=20,

          validation_data=(x_val, y_val))