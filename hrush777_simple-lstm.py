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
import numpy as np

import pandas as pd
import json

from pprint import pprint



with open('../input/train.json') as f:

    data = json.load(f)
_id = list()

cuisine = list()

ingre = list()



for i in range(len(data)):

    _id.append(data[i]['id'])

    cuisine.append(data[i]['cuisine'])

    ingre.append(data[i]['ingredients'])
all_in = list()



for i in ingre:

    for j in i:

        all_in.append(j)
vocab = set(all_in)
word_to_index = dict()

index_to_word = dict()

index = 1



for i in vocab:

    word_to_index[i] = index

    index_to_word[index] = i

    index += 1
seq = list()



for i in ingre:

    list_ = list()

    for j in i:

        x = word_to_index.get(j,-1)

        if x != -1:

            list_.append(x)

    seq.append(list_)
m = 0



for i in range(len(seq)):

    if m < len(seq[i]):

        m = len(seq[i])
m
from keras.preprocessing.sequence import pad_sequences

seq = pad_sequences(seq, maxlen=65)
seq = np.array(seq)
cuisine = pd.get_dummies(cuisine)
cuisine.shape
with open('../input/test.json') as f:

    test_data = json.load(f)
test_id = list()

# test_cuisine = list()

test_ingre = list()



for i in range(len(test_data)):

    test_id.append(test_data[i]['id'])

#     test_cuisine.append(test_data[i]['cuisine'])

    test_ingre.append(test_data[i]['ingredients'])
test_seq = list()



for i in test_ingre:

    list_ = list()

    for j in i:

        x = word_to_index.get(j,-1)

        if x != -1:

            list_.append(x)

    test_seq.append(list_)
test_seq = pad_sequences(test_seq, maxlen=65)
from keras import Sequential

from keras.layers import Embedding, LSTM, Dense, Dropout

embedding_size=64

model=Sequential()

model.add(Embedding(len(vocab)+1, embedding_size, input_length=65))

model.add(LSTM(128, return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(128))

model.add(Dense(20, activation='softmax'))

print(model.summary())
from keras.callbacks import ModelCheckpoint



# checkpoint

filepath="weights-{epoch:02d}-{val_acc:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]





model.compile(loss='categorical_crossentropy', 

             optimizer='adam', 

             metrics=['accuracy'])
model.fit(seq, cuisine, batch_size=64, epochs=15)
pred = model.predict(seq[1].reshape(-1,65))
np.argmax(pred)
cuisine.head()
names = list(cuisine.keys())

names[np.argmax(pred)]
data[1]
test_pred = list()



for i in test_seq:

    pred = model.predict(i.reshape(-1,65))

    test_pred.append(names[np.argmax(pred)])
test_pred = np.array(test_pred).reshape(-1,1)

test_id = np.array(test_id).reshape(-1,1)
output = np.array(np.concatenate((test_id, test_pred),axis=1))
output[0]
output = pd.DataFrame(output,columns = ["id","cuisine"])
output.to_csv('out.csv',index = False)