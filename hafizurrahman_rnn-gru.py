import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt



print(os.listdir("../input"))
from keras import Sequential

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import *

from keras.models import Sequential,Model

from keras.layers import *



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



test_data = test['audio_embedding'].tolist()

x_test = pad_sequences(test_data, maxlen=10)
model = Sequential()

model.add(BatchNormalization(input_shape=(10, 128)))

model.add(Bidirectional(GRU(128, dropout=0.3, recurrent_dropout=0.3, activation='relu', return_sequences=True)))

#model.add(Attention(10))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
#fit on a portion of the training data, and validate on the rest

history = model.fit(x_train, y_train,

          batch_size=300,

          epochs=50,

          validation_data=(x_val, y_val))
y_test = model.predict_classes(x_test, verbose=1)



submission = pd.DataFrame({'vid_id': test['vid_id'].values, 'is_turkey': list(y_test.flatten())})
submission.to_csv("submission.csv", index=False)