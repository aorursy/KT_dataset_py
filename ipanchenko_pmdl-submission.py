from keras.layers import Input, Dense, CuDNNLSTM, AveragePooling1D, TimeDistributed, Flatten, Bidirectional

from keras.models import Model

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



THRESHOLD = 73 

INPUT_WIDTH = 19

N_FEATURES = 22
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.shape, test_df.shape
# convert to float 

train_df.iloc[:, 1:] = train_df.iloc[:, 1:].astype(np.float32)

test_df.iloc[:, 1:] = test_df.iloc[:, 1:].astype(np.float32)



# drop nan-s on Ref column

train_df = train_df.dropna(axis=0, subset=['Ref']).reset_index(drop=True)



# fill na with 0

train_df = train_df.fillna(0.0)

test_df = test_df.fillna(0.0)



# remove outliers

train_df = train_df[train_df.Expected < THRESHOLD]
train_gp = train_df.groupby("Id")

test_gp = test_df.groupby("Id")



test_size = test_df.Id.nunique()

train_size = train_df.Id.nunique()



X_train = np.zeros((train_size, INPUT_WIDTH, N_FEATURES), dtype=np.float32)

X_test = np.zeros((test_size, INPUT_WIDTH, N_FEATURES), dtype=np.float32)



y_train = np.zeros(train_size, dtype=np.float32)



seq_len_train = np.zeros(train_size, dtype=np.float32)

seq_len_test = np.zeros(test_size, dtype=np.float32)
for i, (_, group) in enumerate(train_gp):

    X = group.values

    seq_len = X.shape[0]

    X_train[i,:seq_len,:] = X[:,1:23]

    y_train[i] = X[0,23]

    seq_len_train[i] = seq_len

    

for i, (_, group) in enumerate(test_gp):

    X = group.values

    seq_len = X.shape[0]

    X_test[i,:seq_len,:] = X[:,1:23]

    seq_len_test[i] = seq_len



X_train.shape, y_train.shape, X_test.shape
def get_model(shape):

    inp = Input(shape)

    x = Dense(16)(inp)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = TimeDistributed(Dense(64))(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x = TimeDistributed(Dense(1))(x)

    x = AveragePooling1D()(x)

    x = Flatten()(x)

    x = Dense(1)(x)



    model = Model(inp, x)

    return model
model = get_model((19,22))

model.compile(optimizer='adadelta', loss='mae')

model.summary()
params = {

    'batch_size': 1024,

    'epochs': 20, 

    'validation_split': 0.2

}



model.fit(X_train, y_train, **params)
y_pred = model.predict(X_test)

submission = pd.DataFrame({'Id': np.array(test_df.Id.unique()), 'Expected': y_pred.reshape(-1)})

submission.to_csv('submission.csv', index=False)