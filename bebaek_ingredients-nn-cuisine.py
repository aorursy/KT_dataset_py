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
data = pd.read_json('../input/train.json')
data.head()
# shuffle and split data
n_data = len(data.id)
r_train = 1
r_eval = 0

data = data.sample(frac=1).reset_index(drop=True)
i_train = int(n_data*r_train)
data_train = data[:i_train]
i_eval = int(n_data*(1-r_eval))
data_eval = data[i_eval:]
print('n_train, n_eval = {}, {}'.format(len(data_train), len(data_eval)))
# Build integer-based categorical data

"""Brute-force vectorization. Replaced with Tokenizer API.
import keras

def get_unique_x(lofl):
    x_all = set({})
    for x in lofl:
        x_all = x_all | set(x)
    return np.array(list(x_all))

get_unique_y = lambda l: np.array(list(set(l)))

def get_index(text, dic):
#def get_index(text):
    try:
        return list(dic).index(text)
    except:
        return -1

#v_get_index = np.vectorize(get_index, excluded=['dic'])
v_get_index = np.vectorize(get_index)
v_get_index.excluded.add(1)

# get all x, y
x_train_name = list(data_train.ingredients.values)
y_train_name = list(data_train.cuisine.values)
x_eval_name = list(data_eval.ingredients.values)
y_eval_name = list(data_eval.cuisine.values)

# compile x, y train
x_dic = get_unique_x(x_train_name)
print('All ingredients:', x_dic)
y_dic = get_unique_y(y_train_name)
print('All cuisines:', y_dic)

# vectorize x train
x_train = np.zeros((len(x_train_name), len(x_dic)))
for i, xi in enumerate(x_train):
    xvi = v_get_index(x_train_name[i], x_dic)
    mat = keras.utils.to_categorical(xvi, num_classes=len(x_dic))
    x_train[i] = np.sum(mat, axis=0)
    if i%1000 == 0:
        print(i, xvi)
        print('num of total, 0, 1: {}, {}, {}'.format(len(x_train[i]), len(x_train[i][x_train[i]==0]),
                                                     len(x_train[i][x_train[i]==1])))
print('Shape x_train:', x_train.shape)
input_dim = x_train.shape[1]

# vectorize y train
y_train = keras.utils.to_categorical(v_get_index(y_train_name, y_dic), num_classes=len(y_dic))
print('Shape y_train:', y_train.shape)
output_dim = y_train.shape[1]

# vectorize x eval
x_eval = np.zeros((len(x_eval_name), len(x_dic)))
for i, xi in enumerate(x_eval):
    xvi = v_get_index(x_eval_name[i], x_dic)
    mat = keras.utils.to_categorical(xvi, num_classes=len(x_dic))
    x_eval[i] = np.sum(mat, axis=0)
    if i%1000 == 0:
        print(i, xvi)
        print('num of total, 0, 1: {}, {}, {}'.format(len(x_eval[i]), len(x_eval[i][x_eval[i]==0]),
                                                     len(x_eval[i][x_eval[i]==1])))
print('Shape x_eval:', x_eval.shape)

# vectorize y test
y_eval = keras.utils.to_categorical(v_get_index(y_eval_name, y_dic), num_classes=len(y_dic))
print('Shape y_eval:', y_eval.shape)
"""

from keras.preprocessing.text import Tokenizer

# get all x, y
x_train_name = data_train.ingredients.values
y_train_name = data_train.cuisine.values
x_eval_name = data_eval.ingredients.values
y_eval_name = data_eval.cuisine.values

# vectorize train data
tx = Tokenizer(filters='', split=None)
tx.fit_on_texts(x_train_name)
x_train = tx.texts_to_matrix(x_train_name)
x_eval = tx.texts_to_matrix(x_eval_name)
input_dim = x_train.shape[1]

ty = Tokenizer(filters='', split=None)
ty.fit_on_texts(y_train_name)
y_train = ty.texts_to_matrix(y_train_name)
y_eval = ty.texts_to_matrix(y_eval_name)
output_dim = y_train.shape[1]
y_inv = dict(map(reversed, ty.word_index.items()))

# check
print('Shape x_train, x_eval =', x_train.shape, x_eval.shape)
print('Shape y_train, y_eval =', y_train.shape, y_eval.shape)
i = 0
print('x_train_name[{}] = {}'.format(i, x_train_name[i]))
print('x_train[{}]: N_total = {}, N_0 = {}, N_1 = {}'.format(
    i, len(x_train[i]), len(x_train[i][x_train[i]==0]), len(x_train[i][x_train[i]==1])))
print('y_train_name[{}] = {}'.format(i, y_train_name[i]))
print('y_train[{}]: N_total = {}, N_0 = {}, N_1 = {}'.format(
    i, len(y_train[i]), len(y_train[i][y_train[i]==0]), len(y_train[i][y_train[i]==1])))
print('y_inv =', y_inv)
# Train
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(256, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=40, batch_size=128)
if r_eval > 0:
    score = model.evaluate(x_eval, y_eval, batch_size=128)
    print(model.metrics_names, '=', score)

test = pd.read_json('../input/test.json')
test.head()
# predict
x_test = tx.texts_to_matrix(test.ingredients)
y_test = model.predict(x_test)
y_test_name = [y_inv[k] for k in np.argmax(y_test, axis=1)]

# save
out = pd.DataFrame({
    'id': test.id,
    'cuisine': y_test_name })
out.to_csv('submission.csv', index=False)
out.head()
