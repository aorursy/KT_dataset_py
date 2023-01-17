import os

import random

import re

import spacy

import numpy as np

import itertools

from tqdm import tqdm

from keras.utils import to_categorical

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Embedding, LSTM, SimpleRNN, Bidirectional

from keras.preprocessing.sequence import pad_sequences
def read_imdb(data_path, folder):

    result = []

    for label in ['pos', 'neg']:

        folder_path = os.path.join(data_path, folder, label)

        for file in tqdm(os.listdir(folder_path)):

            with open(os.path.join(folder_path, file), 'r', encoding='utf8') as f:

                text = f.read()

                text = re.sub(r'\s*<br /><br />\s*', ' ', text)

            result.append([text, int(label == 'pos')])

    random.shuffle(result)

    return result



data_path = '../input/aclimdb/aclImdb'

train_data = read_imdb(data_path, 'train')

test_data = read_imdb(data_path, 'test')
def tokenize(texts):

    """

    @param texts: 训练文本构成的列表，每个元素都是一个训练文本

    @return: 分词结果，二维列表

    """

    result = []

    nlp = spacy.load('en_core_web_sm')

    tokenizer = nlp.tokenizer

    for text in tqdm(texts):

        doc = tokenizer(text)

        text = [token.text.lower() for token in doc]

        result.append(text)

    return result



train_X = tokenize([text for text, _ in train_data])

train_y = [label for _, label in train_data]

test_X = tokenize([text for text, _ in test_data])

test_y = [label for _, label in test_data]
def get_vocab(texts):

    tokens = set(itertools.chain(*texts))

    idx2token = ['<pad>', '<unk>'] + list(tokens)

    token2idx = {token: idx for idx, token in enumerate(idx2token)}

    return idx2token, token2idx



idx2token, token2idx = get_vocab(train_X)

print('Vocab size: %d' % len(idx2token))
def create_model(vocab_size, embed_size, hidden_size):

    model = Sequential()

    model.add(Embedding(vocab_size, embed_size))

    model.add(SimpleRNN(hidden_size))

    model.add(Dense(2, activation='softmax'))

    model.compile('adam', 'categorical_crossentropy')

    return model
model = create_model(vocab_size=len(token2idx), embed_size=100, hidden_size=64)

model.summary()
def train(model, train_X, train_y, token2idx, maxlen, epochs, batch_size):

    train_X = [[token2idx[token] for token in text] for text in train_X]

    train_X = pad_sequences(train_X, maxlen=maxlen, padding='post', value=token2idx['<pad>'])

    train_y = to_categorical(np.array(train_y), num_classes=2)

    model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs)
def func():

    train_X = [['that', 'movie', 'is', 'good'],

               ['they', 'feel', 'bad'],

               ['that', 'movie', 'is', 'so', 'bad']

              ]

    train_y = [1, 0, 0]

    train_X = [[token2idx[token] for token in text] for text in train_X]

    print(train_X)

    train_X = pad_sequences(train_X, maxlen=5, padding='post', value=token2idx['<pad>'])

    print(train_X)

    train_y = to_categorical(np.array(train_y), num_classes=2)

    print(train_y)

    

func()
train(model, train_X, train_y, token2idx, maxlen=120, epochs=2, batch_size=512)
def test(model, test_X, token2idx, maxlen, batch_size):

    test_X = [[token2idx[token] if token in token2idx else token2idx['<unk>'] for token in text] for text in test_X]

    test_X = pad_sequences(test_X, maxlen=maxlen, padding='post')

    predictions = model.predict_classes(test_X, batch_size=batch_size)

    predictions = predictions.reshape(-1).tolist()

    return predictions
predictions = test(model, test_X, token2idx, maxlen=120, batch_size=512)
def evaluate(labels, predictions):

    TP, FP, FN = 0, 0, 0

    for y, y_hat in zip(labels, predictions):

        TP += (y == 1 and y_hat == 1)

        FP += (y == 0 and y_hat == 1)

        FN += (y == 1 and y_hat == 0)

    P = TP / (TP + FP)

    R = TP / (TP + FN)

    F1 = 2 * P * R / (P + R)

    return P, R, F1
P, R, F1 = evaluate(test_y, predictions)

print('Precision: %.2f%%' % (P * 100))

print('Recall: %.2f%%' % (R * 100))

print('F1: %.2f%%' % (F1 * 100))
def predict(model, token2idx, sentence):

    nlp = spacy.load('en_core_web_sm')

    sentence = [token.text.lower() for token in nlp.tokenizer(sentence)]

    sentence = [token2idx[token] if token in token2idx else token2idx['<unk>'] for token in sentence]

    sentence = np.array(sentence)

    label = model.predict_classes(sentence)

    print('positive' if label[0] else 'negative')
def create_model(vocab_size, embed_size, hidden_size):

    model = Sequential()

    model.add(Embedding(vocab_size, embed_size))

    model.add(SimpleRNN(hidden_size))

    model.add(Dense(2, activation='softmax'))

    model.compile('adam', 'categorical_crossentropy')

    return model



def run(train_X, train_y, test_X, test_y, embed_size, hidden_size, maxlen, epochs, batch_size, token2idx):

    model = create_model(len(token2idx), embed_size, hidden_size)

    model.summary()

    train(model, train_X, train_y, token2idx, maxlen, epochs, batch_size)

    predictions = test(model, test_X, token2idx, maxlen, batch_size)

    P, R, F1 = evaluate(test_y, predictions)

    print('Precision: %.2f%%' % (P * 100))

    print('Recall: %.2f%%' % (R * 100))

    print('F1: %.2f%%' % (F1 * 100))
embed_size = 100

hidden_size = 64

maxlen = 120

epochs = 2

batch_size = 512

run(train_X, train_y, test_X, test_y, embed_size, hidden_size, maxlen, epochs, batch_size, token2idx)