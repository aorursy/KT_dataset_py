# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('./../input/spooky-author-identification/train.zip')

test = pd.read_csv('./../input/spooky-author-identification/test.zip')
train.head()
test.head()
train.shape,test.shape
one_hot = pd.get_dummies(train['author'])
one_hot.head()
X_train = train['text'].values

y = train.join(one_hot)[['EAP','HPL','MWS']].values

X_test = test['text'].values
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
vocab_size = 25000 #Number of unique words to use

max_len = 250 #length of sentence 
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
len(tokenizer.word_index)
tokenized_train = tokenizer.texts_to_sequences(X_train)

tokenized_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(tokenized_train,maxlen=max_len)

X_test = pad_sequences(tokenized_test,maxlen=max_len)
emd_dim = 32
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size,emd_dim,input_length=max_len),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True)),

    tf.keras.layers.GlobalMaxPool1D(),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(3, activation='sigmoid')

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batch_size = 264

epochs = 10

history = model.fit(X_train,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
import matplotlib.pyplot as plt





def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string, 'val_'+string])

    plt.show()
plot_graphs(history,'accuracy')
plot_graphs(history,'loss')