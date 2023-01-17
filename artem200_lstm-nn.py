from tqdm import tqdm_notebook as tqdm

from numpy.random import seed

from tensorflow import set_random_seed

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import GRU

from keras.optimizers import Adam

from numpy import array

from keras.models import load_model

from keras.layers import Embedding

from keras.backend import clear_session

import pandas as pd

import numpy as np

import datetime, time, random

import math
train_threshold = 1099 - 32 * 7

test_threshold = 1099 - 8 * 7



sz = 40



train_data = {}

test_data = {}

out_data = {}

with open('../input/train.csv', 'r') as f:

    lines = f.readlines()[1:]

    for line in tqdm(lines):

        person_id, days = line.strip().split(',')

        train_data[int(person_id)] = []

        test_data[int(person_id)] = []

        out_data[int(person_id)] = []

        prev_day = None

        for day in list(map(int, days.split())):

            if day <= train_threshold or day > train_threshold and prev_day <= train_threshold:

                train_data[int(person_id)].append(day)

            if day <= test_threshold or day > test_threshold and prev_day <= test_threshold:

                test_data[int(person_id)].append(day)

            out_data[int(person_id)].append(day)

            prev_day = day

        if len(train_data[int(person_id)]) < 2 or not(train_data[int(person_id)][-2] <= train_threshold < train_data[int(person_id)][-1]):

            train_data[int(person_id)] = []

        if len(test_data[int(person_id)]) < 2 or not(test_data[int(person_id)][-2] <= test_threshold < test_data[int(person_id)][-1]):

            test_data[int(person_id)] = []

        train_data[int(person_id)] = train_data[int(person_id)][-sz:]

        test_data[int(person_id)] = test_data[int(person_id)][-sz:]

        out_data[int(person_id)] = out_data[int(person_id)][-(sz - 1):]
train_dataset = {}

test_dataset = {}



for person_id in tqdm(list(train_data.keys())):

    new_row = []

    for day in train_data[person_id]:

        day_tr = [0] * 7

        day_tr[(day - 1) % 7] = 1

        new_row.append(day_tr)

    if len(new_row) >= 4:

        while len(new_row) < sz:

            new_row = [[0] * 7] + new_row

        row_len = len(new_row)

        if row_len >= 4:

            if row_len not in train_dataset:

                train_dataset[row_len] = [new_row]

            else:

                train_dataset[row_len].append(new_row)



for person_id in tqdm(list(test_data.keys())):

    new_row = []

    for day in test_data[person_id]:

        day_tr = [0] * 7

        day_tr[(day - 1) % 7] = 1

        new_row.append(day_tr)

    if len(new_row) >= 4:

        while len(new_row) < sz:

            new_row = [[0] * 7] + new_row

        row_len = len(new_row)

        if row_len >= 4:

            if row_len not in test_dataset:

                test_dataset[row_len] = [new_row]

            else:

                test_dataset[row_len].append(new_row)



for row_len in tqdm(list(test_dataset.keys())):

    test_dataset[row_len] = array(test_dataset[row_len])

for row_len in tqdm(list(train_dataset.keys())):

    train_dataset[row_len] = array(train_dataset[row_len])



print("Train num:", sum([len(b) for a, b in train_dataset.items()]))

print("Test num:", sum([len(b) for a, b in test_dataset.items()]))
def batch_generator_all(tX, ty):

    while True:

        yield tX, ty

def batch_generator(tX, ty, batch_size):

    while True:

        for i in range(tX.shape[0] // batch_size):

            yield tX[i * batch_size: (i + 1) * batch_size], ty[i * batch_size: (i + 1) * batch_size]

row_len = sz

X = train_dataset[row_len][:,:-1].copy().astype('float32')

y = train_dataset[row_len][:,-1]

vX = test_dataset[row_len][:,:-1].copy().astype('float32')

vy = test_dataset[row_len][:,-1]

train_gen = batch_generator(X, y, X.shape[0] // 8)

test_gen = batch_generator(vX, vy, vX.shape[0] // 8)
clear_session()



model = Sequential()

model.add(LSTM(7, activation='tanh', return_sequences=True, dropout = 0.2, input_shape = (sz - 1, 7)))

model.add(LSTM(7, activation='tanh'))

model.add(Dense(7, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer=Adam(0.001), metrics = ['accuracy'])



for i in range(10):

    model.fit_generator(

        generator=train_gen,

        validation_data=test_gen,

        validation_steps=8,

        epochs=10,

        steps_per_epoch=8,

        workers=8,

        use_multiprocessing=False)



    train_win, train_size = 0, 0        

    lstm_result = model.predict(X, batch_size=X.shape[0], verbose=0)

    k = np.argmax(lstm_result, axis=1)

    train_win += (train_dataset[row_len][np.arange(len(k)),-1,k]).sum()

    train_size += len(train_dataset[row_len])



    test_win, test_size = 0, 0

    lstm_result = model.predict(vX, batch_size=vX.shape[0], verbose=0)

    k = np.argmax(lstm_result, axis=1)

    test_win += (test_dataset[row_len][np.arange(len(k)),-1,k]).sum()

    test_size += len(test_dataset[row_len])



    print(datetime.datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S]'), "Iteration #", i + 1, "{:.5f} {:.5f}".format(train_win / train_size, test_win / test_size))
def get_result(out_data, model):

    dataset = {}

    result = {}

    who_is_who = {}

    result = {}

    for person_id in tqdm(list(out_data.keys())):

        new_row = []

        for day in out_data[person_id]:

            day_tr = [0] * 7

            day_tr[(day - 1) % 7] = 1

            new_row.append(day_tr)

        while len(new_row) < sz - 1:

            new_row = [[0] * 7] + new_row

        row_len = len(new_row)

        if row_len < 4:

            result[person_id] = 1

        else:

            if row_len not in dataset:

                dataset[row_len] = [new_row]

                who_is_who[row_len] = [person_id]

            else:

                dataset[row_len].append(new_row)

                who_is_who[row_len].append(person_id)



    for row_len in list(dataset.keys()):

        dataset[row_len] = array(dataset[row_len])



    for row_len in list(dataset.keys()):

        lstm_result = model.predict(dataset[row_len][:, :], batch_size=len(dataset[row_len]), verbose=0)

        k = np.argmax(lstm_result, axis=1)

        for j in range(len(dataset[row_len])):

            result[who_is_who[row_len][j]] = k[j] + 1



    return result
model.save('model.h5')
result = get_result(out_data, model)
def save_solution(fname, result):

    with open(fname, 'w') as f:

        f.write('id,nextvisit\n')

        for person_id in sorted(list(result.keys())):

            f.write(str(person_id) + ', ' + str(result[person_id]) + '\n')

save_solution('solution.csv', result)
print(datetime.datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S]'), "Iteration #", i + 1, "{:.5f} {:.5f}".format(train_win / train_size, test_win / test_size))