# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns



from keras.models import Model

from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils import to_categorical
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.shape, test.shape
train.shape, test.shape
train_length = train.comment_text.apply(len)

train_length.head()
plt.figure(figsize = (12, 5))

plt.hist(train_length, bins = 60, range = [0, 1000], alpha = 0.5, color = 'r')

plt.show()
print("max length : ", np.max(train_length))

print("min length : ", np.min(train_length))

print("mean length : ", np.mean(train_length))

print("75 % percentile : ", np.percentile(train_length, 75))

print("85 % percentile : ", np.percentile(train_length, 85))

print("std length : ", np.std(train_length))
train_length = train.comment_text.apply(lambda x : len(x.split()))

train_length.head()
plt.figure(figsize = (12, 5))

plt.hist(train_length, bins = 60, range = [0, 200], alpha = 0.5, color = 'r')

plt.show()
print("max length : ", np.max(train_length))

print("min length : ", np.min(train_length))

print("mean length : ", np.mean(train_length))

print("75 % percentile : ", np.percentile(train_length, 75))

print("85 % percentile : ", np.percentile(train_length, 85))

print("std length : ", np.std(train_length))
X_train = train['comment_text'].astype(str)

X_test = test['comment_text'].astype(str)
y = np.where(train['target'] >= 0.5, True, False) * 1
y[:10]
num_words = 20000

max_len = 150

emb_size = 128
tok = Tokenizer(num_words = num_words)

tok.fit_on_texts(list(X_train))
X = tok.texts_to_sequences(X_train)

test = tok.texts_to_sequences(X_test)
X = sequence.pad_sequences(X, maxlen = max_len)

X_test = sequence.pad_sequences(test, maxlen = max_len)
X[0]
def model():

    inp = Input(shape = (max_len, ))

    layer = Embedding(num_words, emb_size)(inp)

    layer = Bidirectional(LSTM(50, return_sequences = True, recurrent_dropout = 0.15))(layer)

    layer = GlobalMaxPool1D()(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(50, activation = 'relu')(layer)

    layer = Dropout(0.2)(layer)

    layer = Dense(1, activation = 'sigmoid')(layer)

    model = Model(inputs = inp, outputs = layer)

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    return model
model = model()

model.summary()
hist = model.fit(X, y, batch_size = 1024, epochs = 2, validation_split = 0.2)
vloss = hist.history['val_loss']

loss = hist.history['loss']



x_len = np.arange(len(loss))



plt.plot(x_len, vloss, marker='.', c='red', label='vloss')

plt.plot(x_len, loss, marker='.', c='blue', label='loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')

plt.grid()

plt.show()
y_test = model.predict(X_test)
y_test[:10]
sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction'] = y_test
sub.head()
sub.to_csv('submission.csv', index=False)