import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from scipy import spatial

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical
from keras.layers.embeddings import Embedding

from keras.layers import Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

from keras.layers import Input, Dense

from keras.models import Sequential
from sklearn.metrics import classification_report

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE
train = pd.read_csv("../input/atis-airlinetravelinformationsystem/atis_intents_train.csv", header=None)

test = pd.read_csv("../input/atis-airlinetravelinformationsystem/atis_intents_test.csv", header=None)
words = set(stopwords.words("english"))
words
train.head()
test.head()
train['text'] = train[1].apply(lambda x: ' '.join([word for word in x.split() if word not in (words)]))

test['text'] = test[1].apply(lambda x: ' '.join([word for word in x.split() if word not in (words)]))
train['text'] = train['text'].str.replace('\d+', '')

test['text'] = test['text'].str.replace('\d+', '')
train
text = train['text']

labels = train[0]

test_text = test['text']

test_labels = test[0]
labels.nunique()
from keras.preprocessing.text import Tokenizer

tok = Tokenizer()

tok.fit_on_texts(text)

word_index = tok.word_index
word_index
max_vocab_size = len(word_index) + 1

input_length = 25
train_data_tokens = tok.texts_to_sequences(text)

test_data_tokens = tok.texts_to_sequences(test_text)
train_data_tokens
train_input = pad_sequences(train_data_tokens, input_length)

test_input = pad_sequences(test_data_tokens, input_length)
train_input
label_transformer = preprocessing.LabelEncoder()

label_transformer.fit(labels)
# from sklearn.externals import joblib

# joblib.dump(label_transformer, 'atis-airlinetravelinformationsystem/label_encoder.pk1')

labels = label_transformer.transform(labels)

test_labels = label_transformer.transform(test_labels)
labels
labels = to_categorical(np.asarray(labels))

test_labels = to_categorical(np.asarray(test_labels))
labels
X_train, X_val, y_train, y_val = train_test_split(train_input, labels, test_size=0.2, random_state=1)
X_train
embedded_dim = 300

embedded_index = dict()



with open('../input/glove42b300dtxt/glove.42B.300d.txt', 'r', encoding='utf-8') as glove:

    for line in glove:

        values = line.split()

        word = values[0]

        vector = np.asarray(values[1:], dtype='float32')

        embedded_index[word] = vector
glove.close
embedded_matrix = np.zeros((max_vocab_size, embedded_dim))

for x, i in word_index.items():

    vector = embedded_index.get(x)

    if vector is not None:

        embedded_matrix[i] = vector
model = Sequential()

model.add(Embedding(max_vocab_size, 300, input_length=input_length, weights=[embedded_matrix], trainable=False))
model.add(Conv1D(filters=32, kernel_size=8, activation='selu'))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(10, activation='selu'))

model.add(Dense(8, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, verbose=2)
model.evaluate(X_val, y_val)
def acc(y_true, y_pred):

    return np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)).mean()
predictions = model.predict(test_input)
print(acc(test_labels, predictions))