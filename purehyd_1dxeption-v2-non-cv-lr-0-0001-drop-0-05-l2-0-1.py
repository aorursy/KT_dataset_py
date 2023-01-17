dropout_rate = 0.05

l2_prefct = 0.1

learning_rate = 0.0001

train_epochs = 250

%matplotlib inline



from os import listdir, makedirs

from os.path import isfile, join, basename, splitext, isfile, exists



import numpy as np

import pandas as pd



from tqdm import tqdm_notebook



import tensorflow as tf

import keras.backend as K



import keras

from keras.engine.input_layer import Input



import matplotlib.pyplot as plt

import seaborn as sns



import random, os, sys

from keras.models import *

from keras.layers import *

from keras.callbacks import *

from keras.optimizers import *

from keras.regularizers import *

from keras.initializers import *

import tensorflow as tf

from keras.engine.topology import Layer



pd.set_option('precision', 30)

np.set_printoptions(precision = 30)



np.random.seed(368)

tf.set_random_seed(368)
%%time

train_df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int8, 'time_to_failure': np.float32})
train_df.head()
X_train = train_df.acoustic_data.values

y_train = train_df.time_to_failure.values
ends_mask = np.less(y_train[:-1], y_train[1:])

segment_ends = np.nonzero(ends_mask)



train_segments = []

start = 0

for end in segment_ends[0]:

    train_segments.append((start, end))

    start = end

    

print(train_segments)
plt.title('Segment sizes')

_ = plt.bar(np.arange(len(train_segments)), [ s[1] - s[0] for s in train_segments])
class EarthQuakeRandom(keras.utils.Sequence):



    def __init__(self, x, y, x_mean, x_std, segments, ts_length, batch_size, steps_per_epoch):

        self.x = x

        self.y = y

        self.segments = segments

        self.ts_length = ts_length

        self.batch_size = batch_size

        self.steps_per_epoch = steps_per_epoch

        self.segments_size = np.array([s[1] - s[0] for s in segments])

        self.segments_p = self.segments_size / self.segments_size.sum()

        self.x_mean = x_mean

        self.x_std = x_std



    def get_batch_size(self):

        return self.batch_size



    def get_ts_length(self):

        return self.ts_length



    def get_segments(self):

        return self.segments



    def get_segments_p(self):

        return self.segments_p



    def get_segments_size(self):

        return self.segments_size



    def __len__(self):

        return self.steps_per_epoch



    def __getitem__(self, idx):

        segment_index = np.random.choice(range(len(self.segments)), p=self.segments_p)

        segment = self.segments[segment_index]

        end_indexes = np.random.randint(segment[0] + self.ts_length, segment[1], size=self.batch_size)



        x_batch = np.empty((self.batch_size, self.ts_length))

        y_batch = np.empty(self.batch_size, )



        for i, end in enumerate(end_indexes):

            x_batch[i, :] = self.x[end - self.ts_length: end]

            y_batch[i] = self.y[end - 1]

            

        x_batch = (x_batch - self.x_mean)/self.x_std



        return np.expand_dims(x_batch, axis=2), y_batch
t_segments = [train_segments[i] for i in [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

v_segments = [train_segments[i] for i in [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
x_sum = 0.

count = 0



for s in t_segments:

    x_sum += X_train[s[0]:s[1]].sum()

    count += (s[1] - s[0])



X_train_mean = x_sum/count



x2_sum = 0.

for s in t_segments:

    x2_sum += np.power(X_train[s[0]:s[1]] - X_train_mean, 2).sum()



X_train_std =  np.sqrt(x2_sum/count)



print(X_train_mean, X_train_std)
train_gen = EarthQuakeRandom(

    x = X_train, 

    y = y_train,

    x_mean = X_train_mean, 

    x_std = X_train_std,

    segments = t_segments,

    ts_length = 150000,

    batch_size = 16,

    steps_per_epoch = 400

)



valid_gen = EarthQuakeRandom(

    x = X_train, 

    y = y_train,

    x_mean = X_train_mean, 

    x_std = X_train_std,

    segments = v_segments,

    ts_length = 150000,

    batch_size = 16,

    steps_per_epoch = 400

)
print(train_gen)
def Xeption_1D_Model(dropout_rate):

    i = Input(shape = (150000,1))

    

    x = i

    for ii in range(3):

        x = Convolution1D(64, 7, strides = 4, activation='relu', padding='valid')(x)

        x = BatchNormalization()(x)

        x = Dropout(dropout_rate)(x)



    #return Model(inputs = [i], outputs = [x])

    

    residual = Convolution1D(64, 1, strides = 4, padding='same')(x)

    residual = Dropout(dropout_rate)(residual)

    residual = BatchNormalization()(residual)

    

    x = Convolution1D(64, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv1D(64, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)



    x = MaxPooling1D(7, strides=4, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = keras.layers.add([x, residual])

    

    #return Model(inputs = [i], outputs = [x])

    

    residual = Convolution1D(128, 1, strides = 4, padding='same')(x)

    residual = Dropout(dropout_rate)(residual)

    residual = BatchNormalization()(residual)

    

    x = Convolution1D(128, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv1D(128, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)



    x = MaxPooling1D(7, strides=4, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = keras.layers.add([x, residual])

    

    #return Model(inputs = [i], outputs = [x])

    

    

    for ii in range(8):

        residual = Convolution1D(192, 1, strides = 1, padding='same')(x)

        residual = Dropout(dropout_rate)(residual)

        residual = BatchNormalization()(residual)

        

        x = Activation('relu')(x)

        x = SeparableConv1D(192, 7, strides = 1, padding='same')(x)

        x = Dropout(dropout_rate)(x)

        x = BatchNormalization()(x)

        

        x = Activation('relu')(x)

        x = SeparableConv1D(192, 7, strides = 1, padding='same')(x)

        x = Dropout(dropout_rate)(x)

        x = BatchNormalization()(x)

        

        x = Activation('relu')(x)

        x = SeparableConv1D(192, 7, strides = 1, padding='same')(x)

        x = Dropout(dropout_rate)(x)

        x = BatchNormalization()(x)

        x = keras.layers.add([x, residual])

        

    #return Model(inputs = [i], outputs = [x])

    

    residual = Convolution1D(384, 1, strides = 4, padding='same')(x)

    residual = Dropout(dropout_rate)(residual)

    residual = BatchNormalization()(residual)

    

    x = SeparableConv1D(384, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv1D(384, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)



    x = MaxPooling1D(7, strides=4, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = keras.layers.add([x, residual])

    

    #return Model(inputs = [i], outputs = [x])

    

    residual = Convolution1D(512, 1, strides = 4, padding='same')(x)

    residual = Dropout(dropout_rate)(residual)

    residual = BatchNormalization()(residual)

    

    x = SeparableConv1D(512, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv1D(512, 7, strides = 1, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)



    x = MaxPooling1D(7, strides=4, padding='same')(x)

    x = Dropout(dropout_rate)(x)

    x = keras.layers.add([x, residual])

    

    #return Model(inputs = [i], outputs = [x])

    

    x = GlobalAveragePooling1D()(x)

    x = Dense(1000, activation='relu')(x)

    x = Dropout(dropout_rate)(x)



    x = Dense(1000, activation='relu')(x)

    x = Dropout(dropout_rate)(x)



    x = Dense(1)(x)

    

    return Model(inputs = [i], outputs = [x])
model = Xeption_1D_Model(dropout_rate)

for layer in model.layers:

    layer.W_regularizer = l2(l2_prefct)

#adam = Adam(lr=0.001)

adam = Adam(lr=learning_rate)



model.compile(loss='mean_absolute_error', optimizer=adam)

model.summary()
import time

start_time = time.time()

hist = model.fit_generator(

    generator =  train_gen,

    epochs = train_epochs, 

    verbose = 1, 

    validation_data = valid_gen,

    callbacks = [

        EarlyStopping(monitor='val_loss', patience = 400, verbose = 1)

    ]

)

print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

_= plt.legend(['Train', 'Test'], loc='upper left')
import gc

del train_gen

del valid_gen

del X_train

del y_train

del train_df

gc.collect()

model.save_weights('./trained_model.h5', overwrite=True)
def load_test(ts_length = 150000):

    base_dir = '../input/test/'

    test_files = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]



    ts = np.empty([len(test_files), ts_length])

    ids = []

    

    i = 0

    for f in tqdm_notebook(test_files):

        ids.append(splitext(f)[0])

        t_df = pd.read_csv(base_dir + f, dtype={"acoustic_data": np.int8})

        ts[i, :] = t_df['acoustic_data'].values

        i = i + 1



    return ts, ids
test_data, test_ids = load_test()
X_test = ((test_data - X_train_mean)/ X_train_std)

X_test = np.expand_dims(X_test, 2)

X_test.shape
y_pred = model.predict(X_test)
submission_df = pd.DataFrame({'seg_id': test_ids, 'time_to_failure': y_pred[:, 0]})
submission_df.to_csv("submission.csv", index=False)