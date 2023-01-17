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
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, GRU, Embedding,CuDNNGRU

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
dataset = pd.read_csv('../input/hepsiburada.csv')
dataset
target = dataset['Rating'].values.tolist()

data = dataset['Review'].values.tolist()
seperate = int(len(data)/0.8)

x_train , x_test = data[:seperate], data[seperate:]

y_train , y_test = target[:seperate], target[seperate:]
limit = 10000

tokenizer = Tokenizer(num_words =limit)

tokenizer.fit_on_texts(data) ###tokenleştirme#
tokenizer.word_index ### en çok kullanılan 10000 kelimeyi tokenleştirdi.###
x_train_tokens = tokenizer.texts_to_sequences(x_train) ### cümleleri bir token haline dönüştürdük###
x_train[800]
print(x_train_tokens[800])
x_test_tokens = tokenizer.texts_to_sequences(x_test)
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens ]

num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens)

max_tokens = int(max_tokens)

max_tokens
x_train_pad = pad_sequences(x_train_tokens,maxlen = max_tokens)
x_test_pad = pad_sequences(x_test_tokens,maxlen = max_tokens)
x_train_pad[800]
idx = tokenizer.word_index

inverse_map = dict(zip(idx.values(), idx.keys()))
def tokens_to_string(tokens):

    words=[inverse_map[token] for token in tokens if token!=0]

    text = ' '.join(words)

    return text
x_train[800]
tokens_to_string(x_train_tokens[800])
model = Sequential()

embedding_size = 50 # her kelimeye karşılık gelen 50 uzunluğundaki vektör
model.add(Embedding(input_dim=limit,output_dim=embedding_size,input_length=max_tokens,name='embedding_layer'))
model.add(CuDNNGRU(units=16,return_sequences=True))

model.add(CuDNNGRU(units=8,return_sequences=True))

model.add(CuDNNGRU(units=4))

model.add(Dense(1,activation='sigmoid'))
optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()
model.fit(x_train_pad, y_train,epochs=5,batch_size=256)