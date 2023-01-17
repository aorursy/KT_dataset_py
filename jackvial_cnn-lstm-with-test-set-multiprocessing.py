import sys
import gc
import time
import os
import logging
from multiprocessing import Pool, current_process
from multiprocessing import log_to_stderr, get_logger
from tqdm import tqdm
from numba import jit

import pyarrow.parquet as pq
import pandas as pd
import numpy as np

import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf

from IPython.display import display, clear_output

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

sns.set_style("whitegrid")
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
%%time 
train_set = pq.read_pandas('../input/train.parquet').to_pandas()
%%time
meta_train = pd.read_csv('../input/metadata_train.csv')
@jit('float32(float32[:,:], int32)')
def feature_extractor(x, n_part=1000):
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part,))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output
x_train = []
y_train = []
for i in tqdm(meta_train.signal_id):
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    y_train.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)
    x_train.append(abs(feature_extractor(train_set.iloc[:, idx].values, n_part=400)))
del train_set; gc.collect()
y_train = np.array(y_train).reshape(-1,)
X_train = np.array(x_train).reshape(-1,x_train[0].shape[0])
def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
n_signals = 1 #So far each instance is one signal. We will diversify them in next step
n_outputs = 1 #Binary Classification
#Build the model
verbose, epochs, batch_size = True, 15, 16
n_steps, n_length = 40, 10
X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
model.save_weights('model1.hdf5')
#%%time
meta_test = pd.read_csv('../input/metadata_test.csv')
def process_chunk(arg):
    start_index = arg['start_index']
    chunk_size = arg['chunk_size']
    
    # Test set indices start at 8712
    test_set_start = 8712
    offset_index = (test_set_start + start_index)
    
    # Column name must be a string
    subset_test = pq.read_pandas('../input/test.parquet', columns=[str(offset_index + j) for j in range(chunk_size)]).to_pandas()    
    x_test = []
    for j in range(chunk_size):
        subset_test_row = subset_test[str(offset_index + j)]
        x_test.append(abs(feature_extractor(subset_test_row.values, n_part=400)))
    return x_test
# Define 21 chunks for processing the test set
# on multiple cpus. I have choosen to process in chunks of 1000 (plus the remainder)
# so as to keep within the kernels memory limit
args = []
for i in range(0, 20000, 1000):
    args.append({
        'start_index': i,
        'chunk_size': 1000
    })
    
# Add a chunk for the remainder
args.append({
    'start_index': 20000,
    'chunk_size': 337
})

n_cpu = processes=os.cpu_count()
print('n_cpu: ', n_cpu)

p = Pool(processes=n_cpu)

# Map the chunk args to the the process_chunk function
x_test_chunks = p.map(process_chunk, args)
print(f"multi processing complete. len: {len(x_test_chunks)}")

p.close()
p.join()
x_test = [item for sublist in x_test_chunks for item in sublist]
x_test = np.array(x_test)
print('x_test.shape: ', x_test.shape)
X_test = x_test.reshape((x_test.shape[0], n_steps, n_length, n_signals))
del x_test_chunks
preds = model.predict(X_test)
preds.shape
threshpreds = (preds>0.5)*1
sub = pd.read_csv('../input/sample_submission.csv')
sub.target = threshpreds

# Gave me an LB score of ~0.450
sub.to_csv('submission.csv',index=False)
check_sub = pd.read_csv('submission.csv')
check_sub.head(20)
