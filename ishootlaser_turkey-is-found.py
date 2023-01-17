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
from keras.layers import Conv1D
from keras.models import Model
from keras.layers import BatchNormalization, Input, Dropout, Dense, MaxPooling1D, Lambda
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.activations import relu
import keras.backend as K
import tensorflow as tf
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
train['duration'] = train['end_time_seconds_youtube_clip'] - train['start_time_seconds_youtube_clip']
test['duration'] = test['end_time_seconds_youtube_clip'] - test['start_time_seconds_youtube_clip']
train.head()
test_max_len = max(test['duration'].values)
print('Max duration is {}'.format(test_max_len))
X_train, X_valid, y_train, y_valid = train_test_split(train['audio_embedding'].values, 
                                                    train['is_turkey'].values, 
                                                    test_size = 0.1,
                                                    random_state = 42)
X_train = pad_sequences([np.array(x) for x in X_train], test_max_len)
X_valid = pad_sequences([np.array(x) for x in X_valid], test_max_len)

X_test = pad_sequences([np.array(x) for x in test['audio_embedding'].values], test_max_len)
test_id = test['vid_id'].values
X_valid.shape
# define model
inp_ = Input(X_test.shape[1:])
filters = [256, 512, 1024, 2048]

inp = Conv1D(filters = 128, kernel_size = 2, activation = 'relu')(inp_)
for f in filters:
    inp = Conv1D(filters = f, kernel_size = 3, activation = 'relu')(inp)
    inp = BatchNormalization()(inp)
inp = Lambda(lambda x: tf.squeeze(x, axis = 1))(inp)

num_neurons = [512, 256, 128, 64]

fc = Dense(1024, activation = 'relu')(inp)
for n in num_neurons:
    fc = Dense(n, activation = 'relu')(fc)
    fc = Dropout(0.4)(fc)
    fc = BatchNormalization()(fc)
fc = Dense(1, activation = 'sigmoid')(fc)
    
model = Model(inp_, fc)
model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
model.summary()
ES = EarlyStopping(monitor = 'val_acc', patience = 0, verbose = 1, restore_best_weights = True)
model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data = (X_valid, y_valid), callbacks = [ES])
prediction = model.predict(X_test)
submission = pd.DataFrame(columns = ['vid_id', 'is_turkey'])
submission['vid_id'] = test_id.tolist()
submission['is_turkey'] = prediction
submission.to_csv('submission.csv', index=False)
submission.head()
