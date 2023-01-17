# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# Any results you write to the current directory are saved as output.



df = pd.read_csv(

    '../input/train.csv',

    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
import tensorflow as tf

import numpy as np



tf.set_random_seed(42)

np.random.seed(42)
class DataTrainTestSplit(object):

    def __init__(self, df, sample_size, split=0.1):

        n_samples = df.shape[0] // sample_size

        perm = np.random.permutation(n_samples)

        test_samples = int(np.floor(n_samples * split))

        self.train_slice = perm[:-test_samples]

        self.test_slice = perm[-test_samples:]

    def train(self):

        return self.train_slice

    def test(self):

        return self.test_slice



from tensorflow import keras



class FeatureGenerator(keras.utils.Sequence):

    sample_size = 150_000

    STRIDES = 4 * 1024

    NDIMS = 512

    STEPS = sample_size // STRIDES



    def __init__(self, df, indices, batch_size=32):

        self.dataframe = df

        self.indices = indices

        self.batch_size = batch_size



    def __len__(self):

        return int((self.indices.shape[0] - 1) / self.batch_size) + 1



    @classmethod

    def generate(self, x):

        """

        Return the applitude of the first NDIMS frequencies

        """

        WINDOW = 4 * 1024

        data = []

        for i in range(FeatureGenerator.STEPS):

            begin = i * FeatureGenerator.STRIDES

            end = begin + WINDOW

            wndn = x[begin:end]

            n = wndn.size

            yf = np.fft.fft(wndn, n) * 1/n

            yf = yf[:FeatureGenerator.NDIMS]

            data.append(2.0 * np.abs(yf))

        return np.stack(data)



        

    def __getitem__(self, index: int):

        s_begin = index * self.batch_size

        s_end = min(s_begin + self.batch_size, self.indices.shape[0])



        X_list = []

        y = np.empty((s_end - s_begin))

        for m in range(s_begin, s_end):

            begin = self.indices[m] * self.sample_size

            end = begin + self.sample_size

            X_list.append(

                FeatureGenerator.generate(self.dataframe['acoustic_data'].values[begin:end])

            )

            y[m - s_begin] = self.dataframe['time_to_failure'].iloc[end - 1]



        X = np.stack(X_list)

        return X, y
import tensorflow as tf



from tensorflow.keras.layers import *

from tensorflow.keras.models import Model, Sequential



def make_block(ndims, inp):

    blocks = []

    for i in range(ndims // 64):

        k = Dense(64)(inp)

        q = Dense(64)(inp)

        kq = Multiply()([k, q])

        v = Dense(64, activation='softmax')(kq)

        blocks.append(v)

    join = Concatenate(axis=2)(blocks)

    ff_sum = Add()([inp, join])

    ff_drop = Dropout(0.1)(ff_sum)

    return ff_drop

    

def make_model():

    inp = Input(shape=(FeatureGenerator.STEPS, FeatureGenerator.NDIMS))

    layer1 = make_block(512, inp)

    layer1_seq = LSTM(256, return_sequences=True)(layer1)

    layer2 = make_block(256, layer1_seq)

    layer3 = LSTM(128)(layer2)

    outp = Dense(1)(layer3)

    model = Model(inp, outp)

    model.compile('adam',

                  loss=['mse'],

                  metrics=['mae'])

    return model



model = make_model()

model.summary()
filepath="model.ckpt.hdf5"

cb_checkpoint = keras.callbacks.ModelCheckpoint(

    filepath, monitor='val_mean_absolute_error',

    save_best_only=True, mode='min')



cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



# cross-validation

for fold in range(5):

    print('Generating model {0}'.format(fold))

    slices = DataTrainTestSplit(df, FeatureGenerator.sample_size, split=0.2)

    model = make_model()

    history = model.fit_generator(FeatureGenerator(df, slices.train(), batch_size=64),

                              validation_data=FeatureGenerator(df, slices.test(), batch_size=64),

                              callbacks=[cb_stop, cb_checkpoint],

                              epochs=100)
%matplotlib inline

import matplotlib.pyplot as plt



plt.figure()

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val_loss')

plt.legend()

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.plot()



plt.figure()

plt.plot(history.history['mean_absolute_error'], label='mae')

plt.plot(history.history['val_mean_absolute_error'], label='val_mae')

plt.legend()

plt.title('Mean Absolute Error')

plt.ylabel('Error')

plt.xlabel('Epoch')

plt.plot()
model.load_weights('model.ckpt.hdf5')

model.evaluate_generator(FeatureGenerator(df, slices.test()))
from tqdm import tqdm_notebook



submission = pd.read_csv(

    '../input/sample_submission.csv', index_col='seg_id', dtype={'time_to_failure': np.float32})



for seg_id in tqdm_notebook(submission.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    X = FeatureGenerator.generate(seg['acoustic_data'].values)

    y = model.predict(X[np.newaxis, :])

    submission.loc[seg_id]['time_to_failure'] = y

submission.to_csv('submission.csv')