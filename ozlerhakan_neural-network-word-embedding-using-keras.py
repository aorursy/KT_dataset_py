# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# keras
import keras

# matplotlib
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/spam_or_not_spam.csv')
df.head()
df.label.value_counts()
df.info()
df.dropna(inplace=True)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def tokenizer_sequences(num_words, X):
    
    # when calling the texts_to_sequences method, only the top num_words are considered while
    # the Tokenizer stores everything in the word_index during fit_on_texts
    tokenizer = Tokenizer(num_words=num_words)
    # From doc: By default, all punctuation is removed, turning the texts into space-separated sequences of words
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    
    return tokenizer, sequences
max_words = 10000 
# for the tokenizer, we configure it to only take into account the 1000 most common words when calling the texts_to_sequences method.

maxlen = 300
# maxlen is the dimension that each email will have a fixed word sequence, in this case each email will be of a 1-d tensor (300,).
tokenizer, sequences = tokenizer_sequences(max_words, df.email.copy())
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# # We will pad all input sequences to have the length of 300. Each email will be the same length of sequence.
X = pad_sequences(sequences, maxlen=maxlen)

y = df.label.copy()

print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', y.shape)
max_words = len(tokenizer.word_index) + 1 # 33672 + 1
# 0 is reserved for padding /no data. The word indexes (i.e. tokenizer.word_index) are 1-offset.
# max_words is the size of the vocabulary, you can think of a book is of max_words pages

embedding_dim = 100 # the dimension of the word dictinory, i.e. this will be 100-dimensional word vector
# you can think of a book that each page has embedding_dim words.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()

# embedding dictionary = 33673 * 100 = 3_367_300 parameters
# we have a 33673 x 100 word vector, Embedding accepts 2D input and returns 3D output as shown in the summary
# input_length = the length of input sequences (i.e. e-mails)
model.add(Embedding(max_words, embedding_dim, input_length=maxlen)) # 33673, 100, input_length=300 = (None, 300,100)
# the activations have shape of (33673, 300, embedding_dim=100)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, y_train.shape
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', color='red', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend();
model.evaluate(X_test, y_test)
# loss value & acc metrics
model.save_weights('pre_trained_model_100D.h5')
# just curiosity, let's explore the shape of the trained word embedding
model.layers[0].get_weights()[0].shape
# (vocabulary len x dimension of word vector) = word vector!
model2 = Sequential()

model2.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model2.add(Flatten())
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.summary()
model2.layers[0].set_weights(model.layers[0].get_weights()) # load
model2.layers[0].trainable = False # freeze
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history2 = model2.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)
from sklearn.metrics import precision_score, recall_score

y_test_pred = np.where(model2.predict(X_test) > .5, 1, 0).reshape(1, -1)[0]
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_test_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_test_pred)))
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', color='red', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend();
# predict the test score
model2.evaluate(X_test, y_test)
# loss value & acc metrics
model2.save_weights('pre_trained_model2100D_dense.h5')
