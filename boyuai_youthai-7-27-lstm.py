import argparse

import itertools

import jieba

import numpy as np

from collections import Counter

from keras.models import Model, Sequential

from keras.layers import InputLayer, Embedding, LSTM, Dense, TimeDistributed

from keras.optimizers import SGD, Adam, Adadelta, RMSprop



def build_vocab(text, vocab_lim):

    word_cnt = Counter(itertools.chain(*text))

    vocab_inv = [x[0] for x in word_cnt.most_common(vocab_lim)]

    vocab = {x: index for index, x in enumerate(vocab_inv)}

    return vocab, vocab_inv





def process_file(file_name, use_char_based_model):

    raw_text = []

    with open(file_name, "r") as f:

        for line in f:

            if (use_char_based_model):

                raw_text.extend([str(ch) for ch in line])

            else:

                raw_text.extend([word for word in jieba.cut(line)])

    return raw_text





def build_matrix(text, vocab, length, step):

    M = []

    for word in text:

        index = vocab.get(word)

        if (index is None):

            M.append(len(vocab))

        else:

            M.append(index)

    num_sentences = len(M) // length

    M = M[: num_sentences * length]

    M = np.array(M)



    X = []

    Y = []

    for i in range(0, len(M) - length, step):

        X.append(M[i : i + length])

        Y.append([[x] for x in M[i + 1 : i + length + 1]])

    return np.array(X), np.array(Y)
seq_length = 11

raw_text = process_file("../input/youthai727/poetry.txt", True)

vocab, vocab_inv = build_vocab(raw_text, 4000)

print(len(vocab), len(vocab_inv))

X, Y = build_matrix(raw_text, vocab, seq_length, seq_length)

print(X.shape)

print(Y.shape)

print(X[0])

print(Y[0])
model = Sequential()

model.add(InputLayer(input_shape=(None, )))

model.add(Embedding(input_dim=len(vocab) + 1, output_dim=64, trainable=True))

model.add(LSTM(units=128, return_sequences=True))

model.add(TimeDistributed(Dense(units=len(vocab) + 1, activation='softmax')))



model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')

model.summary()
model.fit(X, Y, batch_size=512, epochs=50, verbose=1, validation_split=0.1)
st = '明月松间照'

print(st, end='')



vocab_inv.append(' ')



for i in range(200):

    X_sample = np.array([[vocab.get(x, len(vocab)) for x in st]])

    pdt = (-model.predict(X_sample)[:, -1: , :][0][0]).argsort()[:5]

    if vocab_inv[pdt[0]] == '\n' or vocab_inv[pdt[0]] == '，' or vocab_inv[pdt[0]] == '。':

        ch = vocab_inv[pdt[0]]

    else:

        ch = vocab_inv[np.random.choice(pdt)]

    print(ch, end='')

    st = st + ch