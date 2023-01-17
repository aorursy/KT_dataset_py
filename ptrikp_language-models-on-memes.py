# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import argparse
import numpy as np
import scipy
import sklearn.pipeline
from pathlib import Path
from typing import List, Any
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import spacy
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
path = '/kaggle/input/facebook-hateful-meme-dataset/data/'
df_train = pd.read_json(path+ 'train.jsonl', lines=True)
df_dev  =pd.read_json(path+ 'dev.jsonl', lines=True)
df_test  =pd.read_json(path+ 'test.jsonl', lines=True)
df_train.label.value_counts().plot(kind='bar')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train['text'])

word_index = tokenizer.word_index
seq = tokenizer.texts_to_sequences(df_train['text'])
max_len = np.max([len(seq[i]) for i in range(len(seq))])
max_len
X_train = tokenizer.texts_to_sequences(df_train['text'])
X_test = tokenizer.texts_to_sequences(df_dev['text'])
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
le = LabelEncoder()
y_train = le.fit_transform(df_train['label'])
y_test = le.transform(df_dev['label'])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
from keras.models import Sequential
from keras.layers import Embedding, Dropout, Bidirectional,LSTM, GlobalMaxPool1D, Dense, MaxPool1D, Conv1D, Flatten
output_dim = 300
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim, input_length=max_len))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))
model.add(Dropout(0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
def score(test_data):
    from mlxtend.plotting import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    y_pred = model.predict(X_test, batch_size=batch_size)
    print("ROC Score = ", roc_auc_score(df_dev['label'], y_pred[:, 1]))
    cm = confusion_matrix(df_dev['label'], np.argmax(y_pred, 1))
    plot_confusion_matrix(cm)
    plt.show()
    print(classification_report(df_dev['label'],np.argmax(y_pred, 1)))

score(X_test)
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters=100,kernel_size=2, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)
%%time
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('/kaggle/input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary=True)
# Lookup word embeddings for our word index

def w2v(model, word_index):
    emb_weights = np.zeros((len(word_index) + 1, output_dim))
    for word, i in word_index.items(): # dict
        if word in model.index2word: # list
            emb_weights[i] = model[word]
    return emb_weights
%%time
emb_google_weights = w2v(model, word_index)
# Either use emb_google_weights as initial weights and train or make the weights static
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim,
                    # loading in static weights
                    weights=[emb_google_weights], trainable=False, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32,kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)
output_dim = 300
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim,
                    # loading in static weights
                    weights=[emb_google_weights], trainable=False, input_length=max_len))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))
model.add(Dropout(0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)

def w2v_other(file, word_index):
    f = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    index = {}
    for line in f:
        tokens = line.rstrip().split(' ')
        index[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    f.close()
    emb_weights = np.zeros((len(word_index) + 1, output_dim))
    for word, i in word_index.items(): # dict
        vec = index.get(word)
        if vec is not None:
            emb_weights[i] = vec
    return emb_weights
emb_fasttext_weights = w2v_other('/kaggle/input/wikinews300d1mvec/wiki-news-300d-1M.vec', word_index)
# Either use emb_google_weights as initial weights and train or make the weights static
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim,
                    # loading in static weights
                    weights=[emb_fasttext_weights], trainable=False, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32,kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)
output_dim = 300
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim,
                    # loading in static weights
                    weights=[emb_fasttext_weights], trainable=False, input_length=max_len))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))
model.add(Dropout(0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)
%%time
emb_glove_weights = w2v_other('/kaggle/input/glove6b/glove.6B.300d.txt', word_index)
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim,
                    # loading in static weights
                    weights=[emb_glove_weights], trainable=False, input_length=max_len))
model.add(Dropout(0.5))
model.add(Conv1D(filters=32,kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)
output_dim = 300
model = Sequential()
model.add(Embedding(len(word_index)+1,output_dim,
                    # loading in static weights
                    weights=[emb_glove_weights], trainable=False, input_length=max_len))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))
model.add(Dropout(0.25))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])
model.summary()
batch_size = 64
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Testing accuracy: {:.1f}".format(acc * 100))
score(X_test)
