# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D

from keras.layers.embeddings import Embedding

from keras.layers.recurrent import LSTM

from keras.models import Sequential

from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

import collections

import matplotlib.pyplot as plt

import nltk

import os

from sklearn.feature_extraction.text import CountVectorizer
train = pd.read_csv("/kaggle/input/umich-si650-nlp/train.csv")

test = pd.read_csv("/kaggle/input/umich-si650-nlp/test.csv")
x_train =train['sentence']

y_train =train['label']
vectorizer = CountVectorizer()

x_train = vectorizer.fit_transform(train.sentence)

x_test = vectorizer.transform(test.sentence)
word_freq = pd.DataFrame({'Word': vectorizer.get_feature_names(), 'Count': x_train.toarray().sum(axis=0)})

word_freq['Frequency'] = word_freq['Count'] / word_freq['Count'].sum()



word_freq_sort = word_freq.sort_values(by='Frequency', ascending=False)

word_freq_sort.head(10)
maxlen = 0

word_freqs = collections.Counter()

num_recs = 0
f = open("/kaggle/input/umich-si650-nlp/train.csv", "rb")

for line in f:

    sentence = str(line).split(",")[0]

    words = nltk.word_tokenize(sentence.encode().decode("ascii", "ignore").lower())

    if len(words) > maxlen:

        maxlen = len(words)

    for word in words:

        word_freqs[word] += 1

    num_recs += 1

f.close()

print("Max number of words in a sentence: ", maxlen)

print("Number of unqiue words: ", len(word_freqs))
MAX_FEATURES = 1500

MAX_SENTENCE_LENGTH = 40

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2

word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}

word2index["PAD"] = 0

word2index["UNK"] = 1

index2word = {v:k for k, v in word2index.items()}
X = np.empty((num_recs, ), dtype=list)

Xt = []

y = np.zeros((num_recs, ))

i = 0



f = open("/kaggle/input/umich-si650-nlp/train.csv", "r")

next(f)

fixed = []

for line in f:

    line_arr = line.split(",")

    if len(line_arr) > 2:

        line_arr[:len(line_arr)-1] = ["".join(line_arr[:len(line_arr)-1])]

        fixed.append(line_arr)

    else:

        fixed.append(line_arr)



for fline in fixed:

        sentence = fline[0]

        label = int(fline[1])

        words = nltk.word_tokenize(sentence.encode().decode("ascii", "ignore").lower())

        seqs = []

        for word in words:

            if word in word2index:

                seqs.append(word2index[word])

            else:

                seqs.append(word2index["UNK"])

        X[i] = seqs

        y[i] = int(label)

        i += 1

f.close()



# Convert every occurence of None to an empty list



for i in range(0,X.shape[0]):

    if X[i] == None:

        X[i] = ()



X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

# Get shape of X

X.shape
# Splitting our data

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
EMBEDDING_SIZE = 128

HIDDEN_LAYER_SIZE = 64

BATCH_SIZE = 32

NUM_EPOCHS = 10



model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])



history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_test, Y_test))
# Plot loss and accuracy values over time

plt.subplot(211)

plt.title("Accuracy")

plt.plot(history.history["accuracy"], color="g", label="Train")

plt.plot(history.history["val_accuracy"], color="b", label="Validation")

plt.legend(loc="best")



plt.subplot(212)

plt.title("Loss")

plt.plot(history.history["loss"], color="g", label="Train")

plt.plot(history.history["val_loss"], color="b", label="Validation")

plt.legend(loc="best")



plt.tight_layout()

plt.show()
score, acc = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)

print("Test score: %.3f, accuracy: %.3f" % (score, acc))



for i in range(5):

    idx = np.random.randint(len(X_test))

    xtest = X_test[idx].reshape(1, 40)

    ylabel = Y_test[idx]

    ypred = model.predict(xtest)[0][0]

    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])

    print("%.0f - %d - %s" % (ypred, ylabel, sent))