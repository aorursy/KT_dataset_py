from keras.backend import clear_session

from keras.utils import to_categorical

from keras import Sequential

from keras.layers import LSTM, Dense, GRU

from keras.optimizers import Adam
from numpy.random import seed

seed(42)

from tensorflow import set_random_seed

set_random_seed(42)

import random as rn

rn.seed(42)

import os

import pandas as pd

import numpy as np

from tqdm import tqdm
class Config:

    data_path = '../input'

    

config = Config()
data_pd = pd.read_csv(os.path.join(config.data_path, 'train.csv'))

n_clients = data_pd.shape[0]

S = np.zeros((n_clients, 157, 7), dtype=np.uint8)

for i in tqdm(range(n_clients)):

    visits = np.fromstring(data_pd['visits'][i], sep=' ', dtype=np.uint16) - 1

    S[i, np.floor_divide(visits, 7), np.remainder(visits, 7)] = 1

S = np.flip(S, axis=1)



for i in tqdm(range(n_clients)):

    s = S[i].sum(axis=-1)

    n_nonzero = np.count_nonzero(s)

    nonzero = np.nonzero(s)

    S[i, :n_nonzero] = S[i, nonzero]

    S[i, n_nonzero:] = 0

    

S = np.flip(S, axis=1)
X_train = S[:, :156]

y_train = np.zeros((n_clients, 7), dtype=np.uint8)

for i in range(n_clients):

    y_train[i, np.nonzero(S[i, 156])[0][0]] = 1
clear_session()

model = Sequential()

model.add(GRU(32, activation='relu', return_sequences=True, input_shape=(156, 7)))

model.add(GRU(64, activation='relu', return_sequences=True))

model.add(GRU(32, activation='relu'))

model.add(Dense(7, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3), metrics=['acc'])
model.fit(

    X_train,

    y_train,

    batch_size=5000,

    epochs=25

)
res = model.predict(S[:, 1:, :], batch_size=5000)

res_argmax = np.argmax(res, axis=-1) + 1
with open('solution.csv', 'wt') as out:

    out.write('id,nextvisit\n')

    for i in range(n_clients):

        out.write(f'{i+1}, {res_argmax[i]}\n')