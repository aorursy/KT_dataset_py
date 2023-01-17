debug = True
_verbose=1 if debug else 0
def printd(input):
    if debug:
        print(input)
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
import pandas as pd 
data = pd.read_csv('../input/creditcard.csv', sep=',')
printd(data.shape)
printd(data.head())
split = 0.8
msk = np.random.rand(len(data)) < split

# Shuffle the entire data set (applies to both train & test)
data = data.sample(frac=1).reset_index(drop=True)

train_labels = data.loc[msk, data.columns =='Class']
train_data  = data.loc[msk, data.columns !='Class']

test_labels = data.loc[~msk, data.columns =='Class']
test_data  = data.loc[~msk, data.columns !='Class']

printd(test_data.head(1))
printd(test_labels.head(1))
printd(train_data.head(1))
printd(train_labels.head(1))

printd(msk[0:5])
printd(train_data.shape)
printd(test_data.shape)
printd(test_data.shape[0]+train_data.shape[0])
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

printd(train_data.head(1))
printd(test_data.head(1))
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    #keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
    
  model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  return model
model = build_model()
model.summary()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(train_data, train_labels, epochs=500,
                    validation_split=0.2, verbose=_verbose,
                    callbacks=[early_stop])
[loss, mae] = model.evaluate(test_data, test_labels, verbose=_verbose)

test_predictions = model.predict(test_data)

frauds = np.where(test_labels[:]==1)[0]
print(frauds[0:5])
print(test_labels.values[frauds[0:5]])
print(np.around(test_predictions[frauds[0:5]]))
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.