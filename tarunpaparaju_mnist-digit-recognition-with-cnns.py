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
train_data_df = pd.read_csv('../input/train.csv')
train_data_df
train_features = (train_data_df.values[:, 1:785]).reshape((len(train_data_df), 28, 28, 1))
train_targets = train_data_df['label'].values.reshape((len(train_data_df), 1))
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
train_features = train_features*(1.0/255.0)
train_targets = to_categorical(train_targets)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(train_features[:33600], train_targets[:33600], epochs=20, batch_size=128, validation_data=(train_features[33600:], train_targets[33600:]))
test_data_df = pd.read_csv('../input/test.csv')
test_data_df
test_features = test_data_df.values.reshape((len(test_data_df), 28, 28, 1))
test_features = test_features*(1.0/255.0)

predictions = model.predict(test_features)
final_predictions = []

for i in range(0, len(predictions)):
    final_predictions.append(list.index(list(predictions[i]), max(predictions[i])))
ids = [i for i in range(1, len(test_data_df) + 1)]
submission = pd.DataFrame(np.transpose(np.array([ids, final_predictions])))
submission.columns = ['ImageId', 'Label']
submission.head(20)
submission.to_csv('MNIST_preds1.csv', index=False)
