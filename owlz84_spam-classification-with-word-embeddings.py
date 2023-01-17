import numpy as np

import pandas as pd

from string import printable

st = set(printable)

data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",

                   names=["labelStr","text"],

                   skiprows=1,

                   usecols=[0,1],

                   encoding="latin-1")



data["text"] = data["text"].apply(lambda x: ''.join(["" if  i not in st else i for i in x]))

data["label"]=data["labelStr"].apply(lambda x: 1 if x == "spam" else 0)



docs = data["text"].values

labels = data["label"].values



print(len(docs))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

# prepare tokenizer

t = Tokenizer()

t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1

# integer encode the documents

encoded_docs = t.texts_to_sequences(docs)

print(vocab_size)

# pad documents to a max length of 4 words

max_length = 20

padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(len(padded_docs))
# load the whole embedding into memory

embeddings_index = dict()

f = open("../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt")

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs

embedding_matrix = np.zeros((vocab_size, 100))

for word, i in t.word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector



print(embedding_matrix.shape)
from keras.models import Sequential

from keras.layers import Dense, Flatten

from keras.layers.embeddings import Embedding

# define the model



model = Sequential()

model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# summarize the model

print(model.summary())

# split dataset into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=42)
# fit the model

model.fit(X_train, y_train, epochs=10, verbose=0)

# evaluate the model

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: %f' % (accuracy*100))