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
with open("../input/bv00180a.txt", 'r') as file:
    text_file = file.read().lower()
vocab = sorted(set(list(text_file)))
cart_to_id = {c: i for i, c in enumerate(vocab)}
id_to_char = {i: c for i, c in enumerate(vocab)}
x_data = []
y_data = []

maxlen = 40

step = 3

for i in range(0, len(text_file) - maxlen - 1, step):
    x_aux = [cart_to_id[c] for c in text_file[i: i + maxlen]]
    y_aux = cart_to_id[text_file[i + maxlen]]
    
    x_data.append(x_aux)
    y_data.append(y_aux)
from keras.utils import to_categorical
x_data = to_categorical(x_data)
y_data = to_categorical(y_data)
x_data.shape, y_data.shape
from keras.layers import *
from keras.models import *
from keras.optimizers import *
input_node = Input(shape=(maxlen, len(vocab)))
lstm_0 = LSTM(128)(input_node)

output = Dense(len(vocab), activation='softmax')(lstm_0)

modelo = Model(input_node, output)
modelo.summary()
modelo.compile(Adam(lr=0.01), loss='categorical_crossentropy')
modelo.fit(x_data, y_data, epochs=100)
import sys
next_chars = 1000
r = np.random.randint(0, len(text_file) - maxlen - 1)
seed = text_file[r:r+maxlen]
print('Seed inicial: {}'.format(seed))

for i in range(next_chars):
    x_pred = [cart_to_id[c] for c in seed]
    x_pred = np.array(x_pred).reshape(1, len(x_pred))
    x_pred = to_categorical(x_pred, num_classes=67)
    
    y_pred = modelo.predict(x_pred)
    next_char = id_to_char[np.argmax(y_pred)]
    
    sys.stdout.write(next_char)
    
    seed = seed[1:] + next_char