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
import tensorflow as tf



# Fix seeds

np.random.seed(20190411)

tf.set_random_seed(20190411)

df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
from tensorflow import keras



class DataTrainTestSplit(object):

    def __init__(self, df, sample_size, split=0.1):

        n_samples = int(np.floor(df.shape[0] / sample_size))

        perm = np.random.permutation(n_samples)

        test_samples = int(np.floor(n_samples * split))

        self.train_slice = perm[:-test_samples]

        self.test_slice = perm[-test_samples:]

    def train(self):

        return self.train_slice

    def test(self):

        return self.test_slice

        

class Generator(keras.utils.Sequence):

  sample_size = 150_000



  def __init__(self, df, indices, batch_size=32):

    self.dataframe = df

    self.indices = indices

    self.batch_size = batch_size



  def __len__(self):

    return int((self.indices.shape[0] - 1) / self.batch_size) + 1

  

  def __getitem__(self, index: int):

    s_begin = index * self.batch_size

    s_end = min(s_begin + self.batch_size, self.indices.shape[0])



    X = np.empty((s_end - s_begin, self.sample_size, 1))

    y = np.empty((s_end - s_begin))

    for m in range(s_begin, s_end):

        begin = self.indices[m] * self.sample_size

        end = begin + self.sample_size

        X[m - s_begin, :] = self.dataframe['acoustic_data'].values[begin:end].reshape(self.sample_size, 1)

        y[m - s_begin] = self.dataframe['time_to_failure'].iloc[end - 1]



    return X, y

import tensorflow as tf



from tensorflow.keras.layers import *

from tensorflow.keras.models import Model, Sequential



sample = Input(shape=(150_000, 1,))



b0_l1 = Conv1D(10, 100, strides=20)(sample)

b0_l2 = Dense(100, activation='relu')(b0_l1)

b0_l3 = Conv1D(100, 200, strides=100)(b0_l2)

b0_l4 = LSTM(100)(b0_l3)



ttf = Dense(1)(b0_l4)

model = Model(inputs=sample, outputs=ttf)

model.compile('sgd', 'mean_absolute_error')

model.summary()
slices = DataTrainTestSplit(df, Generator.sample_size)



cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



history = model.fit_generator(Generator(df, slices.train(), batch_size=128), validation_data=Generator(df, slices.test()),

                              callbacks=[cb_stop],

                              epochs=100)
%matplotlib inline

import matplotlib.pyplot as plt



plt.figure()

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.plot()



plt.figure()

plt.plot(history.history['val_loss'])

plt.title('Validation loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.plot()



submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={'time_to_failure': np.float32})

# X_test = pd.DataFrame(columns=df.columns, dtype=np.float64, index=submission.index)
from tqdm import tqdm_notebook



for seg_id in tqdm_notebook(submission.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    X = seg['acoustic_data'].values.reshape(1, Generator.sample_size, 1)

    y = model.predict(X)

    submission.loc[seg_id]['time_to_failure'] = y

    
submission.to_csv('submission.csv')