# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn import model_selection

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def padding_seq(data):
    """Pad sequences to sequence with length 10."""
    data = np.array(data)
    to_pad = 10 - data.shape[0]
    return np.pad(data, ((0, to_pad), (0, 0)), 'mean')

def create_x(dataframe):
    X = dataframe['audio_embedding']
    X = [padding_seq(data) for data in X]
    return np.stack(X)

def create_y(dataframe):
    return training_data['is_turkey'].values
training_data = pd.read_json("../input/train.json")
X, y = create_x(training_data), create_y(training_data)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
class GruModel(tf.keras.Model):
    def __init__(self, gru_units=128):
        super(GruModel, self).__init__()
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units, dropout=0.2))
        self.dense1 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        result = self.gru(x)
        return self.dense2(self.dense1(result))

model = GruModel()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=200, validation_data=(X_test, y_test))   
model = GruModel()
X, y = create_x(training_data), create_y(training_data)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=200)
test_data = pd.read_json('../input/test.json')
X = create_x(test_data)
pred = model.predict(X)
submit = test_data[['vid_id']].copy()
submit['is_turkey'] = np.squeeze(pred, axis=-1)
submit.to_csv('result.csv', index=False)