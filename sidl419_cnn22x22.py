# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as ks

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Conv2DTranspose, BatchNormalization, MaxPooling2D

from keras.callbacks import EarlyStopping, ModelCheckpoint

from scipy.signal import convolve2d

from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def make_step(X):

    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X

    return (nbrs_count == 3) | (X & (nbrs_count == 2))
NROW, NCOL = 24, 24

drop = 0.5



def create_model(n_hidden_convs=2, n_hidden_filters=128, kernel_size=5):

    nn = Sequential()

    nn.add(Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu', input_shape=(NROW, NCOL, 1)))

    nn.add(BatchNormalization())

    nn.add(Dropout(0.25))

    for i in range(n_hidden_convs):

        nn.add(Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu'))

        nn.add(BatchNormalization())

        nn.add(Dropout(drop))

    nn.add(Conv2D(1, kernel_size, padding='same', activation='sigmoid'))

    nn.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy', 'mae'])

    return nn
train_df = pd.read_csv('../input/gtrain.csv', index_col='id')

test_df = pd.read_csv('../input/gtest.csv', index_col='id')
train_df1 = train_df[train_df['steps'] == 1].copy()

train_df2 = train_df[train_df['steps'] == 2].copy()

train_df3 = train_df[train_df['steps'] == 3].copy()

train_df4 = train_df[train_df['steps'] == 4].copy()

train_df5 = train_df[train_df['steps'] == 5].copy()
for df in [train_df1, train_df2, train_df3, train_df4, train_df5]:

    df['X'] = df['x_0'].apply(lambda x: [x])



    for t in tqdm_notebook((range(1, 484))):

        df['X'] = df['X'] + df['x_{}'.format(t)].apply(lambda x: [x])



    df['X'] = df['X'].apply(lambda x: np.asarray(x))

    df['X'] = df['X'].apply(lambda x: x.reshape((22, 22)))
for df in [train_df1, train_df2, train_df3, train_df4, train_df5]:

    df['Y'] = df['y_0'].apply(lambda x: [x])



    for t in tqdm_notebook((range(1, 484))):

        df['Y'] = df['Y'] + df['y_{}'.format(t)].apply(lambda x: [x])



    df['Y'] = df['Y'].apply(lambda x: np.asarray(x))

    df['Y'] = df['Y'].apply(lambda x: x.reshape((22, 22)))
for df in [train_df1, train_df2, train_df3, train_df4, train_df5]:

    df.drop(['x_' + str(t) for t in range(484)], axis=1, inplace=True)

    df.drop(['y_' + str(t) for t in range(484)], axis=1, inplace=True)
train_df1_step1 = train_df1.copy()

train_df1_step1['X'] = train_df1_step1['X'].apply(lambda x: make_step(x))

train_df1_step1['steps'] = train_df1_step1['steps'].apply(lambda x: x+1)



train_df1_step2 = train_df1_step1.copy()

train_df1_step2['X'] = train_df1_step2['X'].apply(lambda x: make_step(x))

train_df1_step2['steps'] = train_df1_step2['steps'].apply(lambda x: x+1)



train_df1_step3 = train_df1_step2.copy()

train_df1_step3['X'] = train_df1_step3['X'].apply(lambda x: make_step(x))

train_df1_step3['steps'] = train_df1_step3['steps'].apply(lambda x: x+1)



train_df1_step4 = train_df1_step3.copy()

train_df1_step4['X'] = train_df1_step4['X'].apply(lambda x: make_step(x))

train_df1_step4['steps'] = train_df1_step4['steps'].apply(lambda x: x+1)
train_df2_step1 = train_df2.copy()

train_df2_step1['X'] = train_df2_step1['X'].apply(lambda x: make_step(x))

train_df2_step1['steps'] = train_df2_step1['steps'].apply(lambda x: x+1)



train_df2_step2 = train_df2_step1.copy()

train_df2_step2['X'] = train_df2_step2['X'].apply(lambda x: make_step(x))

train_df2_step2['steps'] = train_df2_step2['steps'].apply(lambda x: x+1)



train_df2_step3 = train_df2_step2.copy()

train_df2_step3['X'] = train_df2_step3['X'].apply(lambda x: make_step(x))

train_df2_step3['steps'] = train_df2_step3['steps'].apply(lambda x: x+1)
train_df3_step1 = train_df3.copy()

train_df3_step1['X'] = train_df3_step1['X'].apply(lambda x: make_step(x))

train_df3_step1['steps'] = train_df3_step1['steps'].apply(lambda x: x+1)



train_df3_step2 = train_df3_step1.copy()

train_df3_step2['X'] = train_df3_step2['X'].apply(lambda x: make_step(x))

train_df3_step2['steps'] = train_df3_step2['steps'].apply(lambda x: x+1)
train_df4_step1 = train_df4.copy()

train_df4_step1['X'] = train_df4_step1['X'].apply(lambda x: make_step(x))

train_df4_step1['steps'] = train_df4_step1['steps'].apply(lambda x: x+1)
def compress(train, edge):

    edge = edge * edge

    train['X'] = train['X'].apply(lambda x: x.flatten() )

    for i in range(edge):

        train['x_' + str(i)] = train['X'].apply(lambda x: x[i])

        

    train['Y'] = train['Y'].apply(lambda x: x.flatten() )

    for i in range(edge):

        train['y_' + str(i)] = train['Y'].apply(lambda x: x[i])

        

    train.drop(['X', 'Y'], axis=1, inplace=True)
concat = [train_df1_step1, train_df1_step2, train_df1_step3, train_df1_step4, 

         train_df2_step1, train_df2_step2, train_df2_step3, 

          train_df3_step1,  train_df3_step2, 

          train_df4_step1]



for train in tqdm_notebook(concat):

    compress(train, 22)
concat = [train_df] + concat

train_df = pd.concat(concat)

train_df.reset_index(drop=True, inplace=True)

train_df.index.name = 'id'
train_df['Y'] = train_df['y_0'].apply(lambda x: [x])



for t in tqdm_notebook((range(1, 484))):

    train_df['Y'] = train_df['Y'] + train_df['y_{}'.format(t)].apply(lambda x: [x])



train_df['Y'] = train_df['Y'].apply(lambda x: np.asarray(x))

train_df['Y'] = train_df['Y'].apply(lambda x: x.reshape((22, 22)))





train_df['X'] = train_df['x_0'].apply(lambda x: [x])



for t in tqdm_notebook((range(1, 484))):

    train_df['X'] = train_df['X'] + train_df['x_{}'.format(t)].apply(lambda x: [x])



train_df['X'] = train_df['X'].apply(lambda x: np.asarray(x))

train_df['X'] = train_df['X'].apply(lambda x: x.reshape((22, 22)))





test_df['X'] = test_df['x_0'].apply(lambda x: [x])



for t in tqdm_notebook((range(1, 484))):

    test_df['X'] = test_df['X'] + test_df['x_{}'.format(t)].apply(lambda x: [x])



test_df['X'] = test_df['X'].apply(lambda x: np.asarray(x))

test_df['X'] = test_df['X'].apply(lambda x: x.reshape((22, 22)))
train_df.drop(['x_' + str(t) for t in range(484)], axis=1, inplace=True)

train_df.drop(['y_' + str(t) for t in range(484)], axis=1, inplace=True)

test_df.drop(['x_' + str(t) for t in range(484)], axis=1, inplace=True)
train_df['X'] = train_df['X'].apply(lambda x: np.pad(x, pad_width=1, mode='wrap'))

train_df['Y'] = train_df['Y'].apply(lambda x: np.pad(x, pad_width=1, mode='wrap'))

test_df['X'] = test_df['X'].apply(lambda x: np.pad(x, pad_width=1, mode='wrap'))
compress(train_df, 24)



test_df['X'] = test_df['X'].apply(lambda x: x.flatten() )

for i in range(576):

    test_df['x_' + str(i)] = test_df['X'].apply(lambda x: x[i])

        

test_df.drop(['X'], axis=1, inplace=True)
train_y = train_df[['steps'] + ['y_' + str(t) for t in range(576)]].copy()

train_df.drop(['y_' + str(t) for t in range(576)], axis=1, inplace=True)
models = []

for delta in range(1, 6):

    model = create_model(n_hidden_convs=6, n_hidden_filters=256)

    es = EarlyStopping(monitor='loss', patience=9, min_delta=0.001)

    

    delta_df = train_df[train_df.steps == delta].iloc[:, 1:].values.reshape(-1, 24, 24, 1)

    delta_y = train_y[train_y.steps == delta].iloc[:, 1:].values.reshape(-1, 24, 24, 1)

    

    model.fit(delta_df, delta_y, batch_size=32, epochs=50, verbose=1, callbacks=[es])

    models.append(model)
submit_df = pd.DataFrame(index=test_df.index, columns=['y_' + str(i) for i in range(484)])
for delta in range(1, 6):

    mod = models[delta-1]

    delta_df = test_df[test_df.steps == delta].iloc[:, 1:].values.reshape(-1, 24, 24, 1)

    submit_df[test_df.steps == delta] = mod.predict(delta_df)[:, 1:23, 1:23, :].reshape(-1, 484).round(0).astype('uint8')
submit_df = submit_df.reset_index(level=['id'])
submit_df.to_csv('cnns_sub_large_edges.csv', index=False)