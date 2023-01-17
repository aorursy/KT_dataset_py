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
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import LSTM, Dense, Bidirectional, Input,Dropout,BatchNormalization, CuDNNGRU, CuDNNLSTM, Flatten

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sample_submission = pd.read_csv('../input/sample_submission.csv')
train_train, train_val = train_test_split(train, random_state = 42)
xtrain = [k for k in train_train['audio_embedding']]
ytrain = train_train['is_turkey'].values

xval = [k for k in train_val['audio_embedding']]
yval = train_val['is_turkey'].values

# Pad the audio features so that all are "10 seconds" long
x_train = pad_sequences(xtrain, maxlen=10)
x_val = pad_sequences(xval, maxlen=10)

y_train = np.asarray(ytrain)
y_val = np.asarray(yval)

print('X_train', x_train.shape)
print('X_val', x_val.shape)
print('y_train', y_train.shape)
print('y_val', y_val.shape)
model = Sequential()
model.add(BatchNormalization(input_shape=(10, 128)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train,
          batch_size=32,
          epochs=50,
          validation_data=(x_val, y_val))
score, acc = model.evaluate(x_val, y_val, batch_size=256)
print('Test accuracy:', acc)
test_data = test['audio_embedding'].tolist()
submission = model.predict(pad_sequences(test_data))
submission = pd.DataFrame({'vid_id':test['vid_id'].values,'is_turkey':[x for y in submission for x in y]})
submission['is_turkey'] = submission.is_turkey
print(submission.head(40))
submission.to_csv('submission.csv', index=False)
