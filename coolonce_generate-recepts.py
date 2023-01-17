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
import pandas as pd

all_recepts = pd.read_csv("../input/recepts/all_recepts.csv", sep="\t")
all_recepts = all_recepts.drop(columns='Unnamed: 0')

f = open('../input/receptxt/text.txt', encoding='UTF-8')

all_recepts = f.read()
print(len(all_recepts)/100)
all_recepts_1 = all_recepts[:500000]

len(all_recepts_1)
from keras.models import Input, Model

from keras.layers import Dense, Dropout

from keras.layers import LSTM

from keras.layers.wrappers import TimeDistributed

import keras.callbacks

import keras.backend as K

import scipy.misc

import json



import os, sys

import re

import PIL

from PIL import ImageDraw



from keras.optimizers import RMSprop, Adam

import random

import numpy as np

import tensorflow as tf

from keras.utils import get_file



from IPython.display import clear_output, Image, display, HTML

try:

    from io import BytesIO

except ImportError:

    from StringIO import StringIO as BytesIO
all_recepts['Инструкции'].head()[0]
all_txt = ''
for rows in all_recepts['Инструкции']:

    all_txt = all_txt + str(rows) + '\r\n'
# training_text = all_recepts['Инструкции'].head()[0]

training_text = all_recepts_1
# f = open('/kaggle/working/text.txt', 'w')

# f.write(training_text)
chars = list(sorted(set(training_text)))

char_to_idx = {ch: idx for idx, ch in enumerate(chars)}

len(chars)
def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):

    input = Input(shape=(None, num_chars), name='input')

    prev = input

    for i in range(num_layers):

        lstm = LSTM(num_nodes, return_sequences=True, name='lstm_layer_%d' % (i + 1))(prev)

        if dropout:

            prev = Dropout(dropout)(lstm)

        else:

            prev = lstm

    dense = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(prev)

    model = Model(inputs=[input], outputs=[dense])

#     optimizer = RMSprop(lr=0.01)

    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
model = char_rnn_model(len(chars), num_layers=2, num_nodes=512)

model.summary()

CHUNK_SIZE = 160



def data_generator(all_text, char_to_idx, batch_size, chunk_size):

    X = np.zeros((batch_size, chunk_size, len(char_to_idx)))

    y = np.zeros((batch_size, chunk_size, len(char_to_idx)))

    while True:

        for row in range(batch_size):

            idx = random.randrange(len(all_text) - chunk_size - 1)

            chunk = np.zeros((chunk_size + 1, len(char_to_idx)))

            for i in range(chunk_size + 1):

                chunk[i, char_to_idx[all_text[idx + i]]] = 1

            X[row, :, :] = chunk[:chunk_size]

            y[row, :, :] = chunk[1:]

        yield X, y



next(data_generator(training_text, char_to_idx, 4, chunk_size=CHUNK_SIZE))
# !pip install tensorflow-gpu
# tf.device('/gpu:0')

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# from tensorflow.python.client import device_lib

# print(device_lib.list_local_devices())

# from keras import backend as K

# K.tensorflow_backend._get_available_gpus()

# assert tf.test.is_gpu_available()

# assert tf.test.is_built_with_cuda()



model = keras.models.load_model('/kaggle/input/models/modelReceptGener.h5')

model.summary()
early = keras.callbacks.EarlyStopping(monitor='loss',

                              min_delta=0.03,

                              patience=3,

                              verbose=0, mode='auto')



BATCH_SIZE = 256

#with tf.device('/device:GPU:0'):

model.fit_generator(

    data_generator(training_text, char_to_idx, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE),

    epochs=10,

    callbacks=[early,],

    steps_per_epoch=2 * len(training_text) / (BATCH_SIZE * CHUNK_SIZE),

    verbose=2

)

model.save('/kaggle/working/modelReceptGener1.h5')
model.save('/kaggle/working/modelReceptGener.h5')

# loss: 1.4710 - accuracy: 0.6877


def generate_output(model, training_text, start_index=None, diversity=None, amount=400):

    if start_index is None:

        start_index = random.randint(0, len(training_text) - CHUNK_SIZE - 1)

#     generated = training_text[start_index: start_index + CHUNK_SIZE]

    generated = "Суп"

    yield generated + '#'

    for i in range(amount):

        x = np.zeros((1, len(generated), len(chars)))

        for t, char in enumerate(generated):

            x[0, t, char_to_idx[char]] = 1.

        preds = model.predict(x, verbose=0)[0]

#         print(preds)

        if diversity is None:

            next_index = np.argmax(preds[len(generated) - 1])

        else:

            preds = np.asarray(preds[len(generated) - 1]).astype('float64')

            preds = np.log(preds) / diversity

            exp_preds = np.exp(preds)

            preds = exp_preds / np.sum(exp_preds)

            probas = np.random.multinomial(1, preds, 1)

            next_index = np.argmax(probas)     

        next_char = chars[next_index]

        yield next_char



        generated += next_char

    return generated



for ch in generate_output(model, training_text, diversity=0.2):

    sys.stdout.write(ch)

print()
new_model = keras.models.load_model('/kaggle/input/models/modelReceptGener.h5')
for ch in generate_output(new_model, training_text, diversity=0.8):

    sys.stdout.write(ch)

print()
new_model
generated = "Рассольник"

    

x = np.zeros((1, len(generated), len(chars)))

for t, char in enumerate(generated):

    x[0, t, char_to_idx[char]] = 1.

preds = model.predict(x, verbose=0)[0]



next_index = np.argmax(preds[len(generated) - 1])

chars[next_index]