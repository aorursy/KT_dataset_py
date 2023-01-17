import pandas as pd

import numpy as np

import keras

from keras import optimizers

from keras import backend as K

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 

from keras.utils import plot_model

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import os, re, csv, math, codecs

from tqdm import tqdm

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt
MAX_NB_WORDS = 100000

max_seq_len = 50

tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('russian'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
df = pd.read_csv('../input/reviews/reviews.csv', sep='\t')
df.head()
df.sentiment.value_counts()
lables = df['sentiment']

train = df['review']

lables = pd.get_dummies(lables)

lables.head()
train_x, test_x, train_y, test_y = train_test_split(train, lables, test_size=0.2)
ft = codecs.open('../input/fasttext-russian-2m/wiki.ru.vec', encoding='utf-8')
embeddings_index = {}

for line in tqdm(ft):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

ft.close()

print('found %s word vectors' % len(embeddings_index))
processed_train = []

for doc in tqdm(train_x):

    tokens = tokenizer.tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    processed_train.append(" ".join(filtered))





processed_test = []

for doc in tqdm(test_x):

    tokens = tokenizer.tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    processed_test.append(" ".join(filtered))



print("tokenizing input data...")

tokenizer = Tokenizer(num_words=MAX_NB_WORDS+1, lower=True, char_level=False)

tokenizer.fit_on_texts(processed_train + processed_test)

word_seq_train = tokenizer.texts_to_sequences(processed_train)

word_seq_test = tokenizer.texts_to_sequences(processed_test)

word_index = tokenizer.word_index

print("dictionary size: ", len(word_index))



#pad sequences

word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)

word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
#training params

batch_size = 250 

num_epochs = 10



#model parameters

num_filters = 40

embed_dim = 300 

weight_decay = 1e-4
words_not_found = []

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words+1, embed_dim))

for word, i in word_index.items():

    if i >= nb_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and len(embedding_vector) > 0:

        embedding_matrix[i] = embedding_vector

    else:

        words_not_found.append(word)

print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


model = Sequential()

model.add(Embedding(nb_words+1, embed_dim,

          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

model.add(MaxPooling1D(2))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Dense(3, activation='softmax'))



adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)

callbacks_list = [early_stopping]
#model training

hist = model.fit(word_seq_train, train_y, batch_size=batch_size,

                 epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)
pred = model.predict(word_seq_test)

pred
pred = np.argmax(pred, axis = 1)

pred
true = test_y.values

true = np.argmax(true, axis = 1)

true
from sklearn.metrics import f1_score

print('F1-Score:', f1_score(true, pred, average='macro'))