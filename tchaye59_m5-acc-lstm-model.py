# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

def log(*args):

    os.system(f'echo \"{args}\"')

    print(*args)
! pip install --upgrade pip

! pip install tensorflow_datasets
import tensorflow as tf

from tensorflow import keras

import pickle

import pandas

import numpy as np

from matplotlib import pyplot as plt

import gc

import threading

import time

import tensorflow_datasets.public_api as tfds

from tensorflow.keras import backend as K

from sklearn import metrics
tf.random.set_seed(1234)

np.random.seed(1234)

# The columns give us an idea of what our data looks like

columns = pickle.load(open('/kaggle/input/data-preparation/columns.pkl','rb'))

max_day = 1941

print(f'Columns-{len(columns)} : {columns}')

print('Target: ',columns[3])

target_col = 3
# This function is just a wrapper to a function that calls add_time_steps

def preprocess(lookback, delay, target_col=target_col ,lookback_step=1,test=False,val_days=0,return_key=False,val=False):

    # It takes as input a single time series data and applies some transformations to add time step

    def fn(inputs):

        # Remember _generate_examples of MyFullDataset we yielded a dictionary containing key and input

        values = inputs['input'] 

        key = inputs['key'] 

        return add_time_steps_ntarget((key,values),

                              lookback,delay,

                              lookback_step=lookback_step,

                              target_col=target_col,

                              val_days=val_days,

                              val=val,

                              test=test,

                              return_key=return_key)

    return fn

   

# This function takes an item with all it 1941 days salles, prices, and calendar data adds lookback 

# and generate the inputs and targets. 

# This function output a single target

def add_time_steps(inputs, lookback, delay,target_col=target_col,test=False,lookback_step=1,val=False,val_days=0,return_key=False):

    key,values = inputs

    max_index = values.shape[0] - 1

    min_index = 0

    y=None

    idx = tf.signal.frame(tf.range(min_index,max_index),lookback,lookback_step)

    if not test:

        idx = idx[tf.reduce_all(idx+delay <= max_index,axis=-1)]

        if val:

            idx = idx[-val_days:]

        else:

            if val_days:

                idx = idx[:-val_days]

        y_idx = idx[...,-1]+delay

        y = tf.gather(values, y_idx)[...,target_col]

    else:

        idx = idx[-delay:]

    X = tf.gather(values, idx)

    if not test and return_key:

        return (key,X,y)

    return (X,y) if not test else (key,X)



# This function takes an item with all it 1941 days salles, prices, and calendar data adds lookback 

# and generate the inputs and targets. 

# this function generates for each input the next 28 forecasts

def add_time_steps_ntarget(inputs, lookback, delay,target_col=target_col,test=False,lookback_step=1,val=False,val_days=0,return_key=False):

    key,values = inputs

    max_index = values.shape[0] - 1

    min_index = 0

    val_steps = val_days//delay

    y=None

    idx = tf.signal.frame(tf.range(min_index,max_index),lookback,lookback_step)

    if not test:

        y_idx = tf.range(delay) + idx[:,-1,tf.newaxis]

        

        select = tf.math.less_equal(y_idx[:,-1],max_index)

        idx = idx[select]

        y_idx = y_idx[select]

        

        if val:

            y_idx = y_idx[-val_steps:]

            idx = idx[-val_steps:]

        else:

            if val_days:

                idx = idx[:-val_steps]

                y_idx = y_idx[:-val_steps]

        y = tf.gather(values[...,target_col], y_idx)

    else:

        idx = idx[-1:]

    X = tf.gather(values, idx)

    if not test and return_key:

        return (key,X,y)

    return (X,y) if not test else (key,X)
# We need again this class

class MyFullDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    

    def _split_generators(self, dl_manager):

        return [

            tfds.core.SplitGenerator(

                    name=f'train',

                    gen_kwargs={

                    },

            )

        ]

    

    def _info(self):

        shape = (max_day,len(columns))

        return tfds.core.DatasetInfo(

            builder=self,

            description=(""),

            features=tfds.features.FeaturesDict({

                "input": tfds.features.Tensor(shape=shape,dtype=tf.float32),

                "key": tfds.features.Tensor(shape=(),dtype=tf.int32),

            }),

        )

    

   

    def _generate_examples(self,**args):

        # We no longer need this function because we already build our dataset

        pass
log("Download dataset")

ds_path = '/kaggle/input/data-preparation/'

builder = MyFullDataset(data_dir=ds_path)

builder.download_and_prepare()



# Amazing right look at your dataset info : we only have one split named train.

# No need to worry we will use our preprocess and build differents pipelines to access the training, validation and test data from this single split

log(builder.info)
log("Try data access")

 

dataset = builder.as_dataset()['train']



for item in  dataset.take(2):

    # We access 2 time-series

    log('key: ', item['key']) 

    log('input: ',item['input'].shape) 



# We apply some transformations: add lookback and get the input and target

lookback = 5

delay = 28

preprocessor = preprocess(lookback, delay)

dataset = dataset.map(preprocessor)

for X,y in  dataset.take(1):

    log(X.shape,y.shape)



del dataset

gc.collect()
val_days = 28 # we use last 28 days of each time step for validation

lookback = 28 # time steps

lookback_step = 1

delay = 28 # We we are forecasting 28 days in the future

batch_size = 2**10

buffer_size= batch_size*100

prefetch = 10 # 

total_num_examples = 30490

ds_name = 'train' # The split name

load_weights = True

train = False

log(batch_size)
# Since our dataset is not normalized, we need  our first layer to be a normalization layer

def build_model():

    

    input_ = keras.layers.Input(shape=(lookback,len(columns),))

    bn = keras.layers.BatchNormalization()(input_)

    

    lstm = keras.layers.Bidirectional(keras.layers.LSTM(256,return_sequences=True,recurrent_dropout=0.1))(bn)

    lstm = keras.layers.Bidirectional(keras.layers.LSTM(256,recurrent_dropout=0.1))(lstm)

    dense = keras.layers.Dense(delay,activation=keras.activations.relu)(lstm)#delay

    dense = keras.layers.Activation('relu')(dense)

    

    model = keras.models.Model(input_,dense)

    log(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(0.01),loss=keras.losses.mean_squared_error, metrics=[])

    return model
model = build_model()

dataset_ = builder.as_dataset()[ds_name].repeat()

# Let Try to load weight from the previous commit

if load_weights and os.path.exists('/kaggle/input/modelh5/model.h5'):

    log('Loading weights')

    model.load_weights('/kaggle/input/modelh5/model.h5')
# This pipeline provides training dataset

preprocessor = preprocess(lookback, delay,lookback_step=lookback_step,val_days=val_days)

train_dataset = dataset_.take(total_num_examples).repeat().shuffle(buffer_size=1000).map(preprocessor).unbatch()

train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(prefetch)

train_steps = (total_num_examples*max_day-val_days-delay)//batch_size//lookback_step

log(train_steps)
# This one provides validation dataset

val_preprocessor = preprocess(lookback, delay,val_days=val_days,val=True,lookback_step=lookback_step)

val_dataset = dataset_.take(total_num_examples).repeat().map(val_preprocessor).unbatch()

val_dataset = val_dataset.batch(total_num_examples//10).prefetch(prefetch)

val_steps = 10

log(val_steps)
if train:

    log('Start training')

    history = model.fit(train_dataset,

                      steps_per_epoch=train_steps,

                      epochs=10,

                      validation_data=val_dataset,

                      validation_steps=val_steps,

                      callbacks=[

                            keras.callbacks.EarlyStopping(

                                monitor='loss',

                                patience=100,

                                restore_best_weights=True,

                            ),

                            keras.callbacks.ModelCheckpoint(

                                filepath=f'model.h5',

                                monitor='val_loss',

                                save_best_only=True,

                                save_weights_only=True,

                            ),

                        ],

                )
if train:

    pd.DataFrame(history.history).plot(figsize=(15,8))
del train_dataset

del val_dataset

gc.collect()
#model.load_weights(f'model.h5')
# Evaluate our validation data with the leaderboard



df_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv',index_col=0)

df_val.iloc[total_num_examples:] = 0 # set evaluation to 0



val_preprocessor = preprocess(lookback, delay,val_days=val_days,val=True,lookback_step=lookback_step)



fn_filter_key = lambda input_ : input_['key']

for keys in dataset_.take(total_num_examples).map(fn_filter_key).batch(total_num_examples):

    print('keys : ',keys.shape)

    keys = keys.numpy()

    keys = np.argsort(keys) # retrieve items ordering



dataset = dataset_.take(total_num_examples)

dataset = dataset.map(val_preprocessor).batch(total_num_examples)



for X,y in dataset:

    log(f'X: {X.shape}, y: {y.shape}, keys: {keys.shape}')

    X = tf.reshape(X,(-1,lookback,len(columns)))#

    y_pred = model.predict(X, batch_size=30490)



df_val.iloc[:total_num_examples,:] = y_pred[keys]

df_val.to_csv('validation_submission.csv')
def generate_submission(model, lookback,lookback_step, delay, out_path='submission.csv'):

    preprocessor = preprocess(lookback, delay,lookback_step=lookback_step,test=True)

    df_sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv',index_col=0)

    df_sub.iloc[:total_num_examples,:] = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv',index_col=0).iloc[:,-28:].values

    

    dataset = dataset_.take(total_num_examples)

    dataset = dataset.map(preprocessor).batch(total_num_examples)

    

    for keys,X in dataset:

        keys = np.argsort(keys.numpy()) # retrieve items ordering

        log('X: ',X.shape)

        X = tf.reshape(X,(-1,lookback,len(columns)))#

        y = model.predict(X, batch_size=30490)

        

    df_sub.iloc[total_num_examples:,:] = y[keys]

    df_sub.to_csv(out_path)

    del dataset
log('Generate Submission')

generate_submission(model,lookback,lookback_step, delay)