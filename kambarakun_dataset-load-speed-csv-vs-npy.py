%time

import numpy as np

import pandas as pd
%time

df_train = pd.read_csv('../input/train.csv')

np_train_x = np.reshape(df_train.values[:, 1:].astype(np.uint8), (df_train.shape[0], 28, 28))

np_train_y = np.reshape(df_train.values[:, 0].astype(np.uint8), (df_train.shape[0], 1))

df_test = pd.read_csv('../input/test.csv')

np_test_x = np.reshape(df_test.values.astype(np.uint8), (df_test.shape[0], 28, 28))
%time

np.save('./np_train_x.npy', np_train_x)

np.save('./np_train_y.npy', np_train_y)

np.save('./np_test_x.npy', np_test_x)
%%timeit -n 50 -r 3 -p 5

df_train = pd.read_csv('../input/train.csv')

np_train_x = np.reshape(df_train.values[:, 1:].astype(np.uint8), (df_train.shape[0], 28, 28))

np_train_y = np.reshape(df_train.values[:, 0].astype(np.uint8), (df_train.shape[0], 1))

df_test = pd.read_csv('../input/test.csv')

np_test_x = np.reshape(df_test.values.astype(np.uint8), (df_test.shape[0], 28, 28))
%%timeit -n 50 -r 3 -p 5

np_train_x = np.load('./np_train_x.npy')

np_train_y = np.load('./np_train_y.npy')

np_test_x = np.load('./np_train_x.npy')