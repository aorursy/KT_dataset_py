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
import tensorflow

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Bidirectional, Embedding, LSTM, Dense

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
f = open("/kaggle/input/imdblabelled-sentences/imdb_labelled.txt", "r")

data = []

label = []

for d in f.readlines():

    arr = d.split("  	")

    data.append(arr[0])

    label.append(arr[1])
#Using keras to tokenize and pad inputs

tokenize = Tokenizer(oov_token="<OOV>")

tokenize.fit_on_texts(data)

word_index = tokenize.word_index

train = tokenize.texts_to_sequences(data)

data = pad_sequences(train, padding="post")



#Getting length of the padded input

maxlen = data.shape[1]

print(maxlen)



#Example of padded inputs

print(data[0])
#Making labels an ndarray as it was a string list

label = np.array(label, dtype ='float64')
vocab_size = len(word_index) + 1



model = Sequential()

model.add(Embedding(vocab_size, 32))

model.add(Bidirectional(LSTM(128, return_sequences=True)))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(1, activation="sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'])

history = model.fit(data, label, epochs=15)
seq = ['Amazingly bad', 'Wonderful','Could be better', 'Did not live up to the excitement']

train = tokenize.texts_to_sequences(seq)

d = pad_sequences(train, padding="post",maxlen = maxlen)

predict = model.predict_classes(d)

for p in predict:

  if p == 1:

      print("Positive")

  else:

      print("Negative")