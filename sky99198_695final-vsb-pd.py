import keras

import keras.backend as K

from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten

from keras.models import Sequential

import tensorflow as tf

import gc

from numba import jit

from IPython.display import display, clear_output

from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import sys

sns.set_style("whitegrid")

print(tf.__version__)

import pyarrow.parquet as pq

import pandas as pd

import numpy as np

import time

from numpy.fft import *

from scipy import fftpack
def low_pass(s, threshold=3e7):

    fourier = rfft(s)

    frequencies = rfftfreq(s.size, d=2e-2/s.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)



def high_pass(s, threshold=100):

    fourier = rfft(s)

    frequencies = rfftfreq(s.size, d=2e-2/s.size)

    fourier[frequencies < threshold] = 0

    return irfft(fourier)



def feature_extractor(data):

    #print("666")

    #print(np.shape(data))

    data=list(data)

    output=[]

    leg=800000//2000

    temp=[data[x:x+leg] for x in range(len(data)//leg)]

    #print(np.shape(temp))

    #ave=np.average(data)

    #output= list(map(lambda x: [max(x),min(x)], temp))

    output= list(map(lambda x: max(x)-min(x), temp))

    return output

'''

def feature_extractor(data):

    #print("666")

    #print(np.shape(data))

    data=list(data)

    output=[]

    temp=[data[x:x+320] for x in range(len(data)//320)]

    #print(np.shape(temp))

    

    output= list(map(lambda x: max(x)-min(x), temp))

    return output



def feature_extractor(data):

    #print("666")

    #print(np.shape(data))

    data=list(data)

    output=[]

    temp=[data[x:x+500] for x in range(len(data)//500)]

    #print(np.shape(temp))

    #temp_max = list(map(lambda x: max(x), temp))

    

    #temp_min = list(map(lambda x: min(x), temp))

    #print(temp_max[1000])

    #print(np.shape(temp_max),np.shape(temp_min))

    #for i in temp:

    #    output.append(np.average(temp))

    #print(len(output))

    output=list(map(lambda x: max(x), temp))

    return output



'''
def divide_test_data(start,end):

    test_metadata = pd.read_csv('../input/metadata_train.csv') 

    test_index_list = test_metadata['signal_id'][start:end].values

    test_index_lists = []

    #print(start,end)

    for i in range(len(test_index_list)):

        test_index_lists.append(str(test_index_list[i]))

    return test_index_lists

    test_index_lists_master.append(test_index_lists)

test_index_lists_master=[]

test_index_lists_master.append(divide_test_data(0,1000))

#test_index_lists_master.append(divide_test_data(500,1000))

test_index_lists_master.append(divide_test_data(1000,2000))

#test_index_lists_master.append(divide_test_data(1500,2000))

test_index_lists_master.append(divide_test_data(2000,3000))

#test_index_lists_master.append(divide_test_data(2500,3000))

test_index_lists_master.append(divide_test_data(3000,4000))

#test_index_lists_master.append(divide_test_data(3500,4000))

test_index_lists_master.append(divide_test_data(4000,5000))

#test_index_lists_master.append(divide_test_data(4500,5000))

test_index_lists_master.append(divide_test_data(5000,6000))

#test_index_lists_master.append(divide_test_data(5500,6000))

test_index_lists_master.append(divide_test_data(6000,7000))

#test_index_lists_master.append(divide_test_data(6500,7000))

test_index_lists_master.append(divide_test_data(7000,8000))

#test_index_lists_master.append(divide_test_data(7500,8000))

test_index_lists_master.append(divide_test_data(8000,8712))

#train_set =np.array(pq.read_pandas(('/home/tfboy/Downloads/vsb-power-line-fault-detection/train.parquet'),columns=test_index_lists_master[0]).to_pandas().T)

#print(np.shape(list(train_set)))

#print(len(train_set[0]))

#print(len(train_set))

meta_train= pd.read_csv('../input/metadata_train.csv')


#meta_train= pd.read_csv('/home/tfboy/Downloads/vsb-power-line-fault-detection/metadata_train.csv')

#train_set = pq.read_pandas('/home/tfboy/Downloads/vsb-power-line-fault-detection/train.parquet').to_pandas()



x_train = []

y_train = []

#for i in tqdm(meta_train.signal_id):

for i in range(len(test_index_lists_master)):

    print(i)

    t1=time.time()

    train_set =np.array(pq.read_pandas(('../input/train.parquet'),columns=test_index_lists_master[i]).to_pandas().T)

    #y_train.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)

    #print(len(train_set))

    train_set2=[]

    #train_set2=list(map(lambda x: expend_data(x), train_set))

    #print(len(train_set[0]),len(train_set2[0]))

    #x_train.append(list(pool.map(feature_extractor, train_set)))

    #x_train.append(list(pool.map(feature_extractor, train_set2)))

    x_train.append(list(map(lambda x: feature_extractor(x), train_set)))

    #x_train.append(list(map(lambda x: feature_extractor(x), train_set2)))

    gc.collect()

    print(time.time()-t1)

    
print(np.shape(x_train))





temp=np.concatenate((x_train[0], x_train[1]), axis=0)

for i in range(2,9):

    print(i)

    temp=np.concatenate((temp, x_train[i]), axis=0)

print(np.shape(temp))

#del train_set; gc.collect()
y_train=[]

for i in tqdm(meta_train.signal_id):

    y_train.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)

    

    



y_train = np.array(y_train).reshape(-1,)

X_train = np.array(temp).reshape(-1,temp[0].shape[0])
def keras_auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc


n_signals = 1 #So far each instance is one signal. We will diversify them in next step

n_outputs = 1 #Binary Classification

#Build the model

verbose, epochs, batch_size = True, 15, 16

n_steps, n_length = 200, 10

print(np.shape(X_train))

X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))

print(np.shape(X_train))
# define model

model = Sequential()

model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))

#model.add(TimeDistributed(Dropout(0.5)))

model.add(TimeDistributed(Conv1D(filters=96, kernel_size=3, activation='relu')))

#model.add(TimeDistributed(Dropout(0.5)))

#model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))

#model.add(TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')))

model.add(TimeDistributed(Dropout(0.5)))

#model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

#model.add(TimeDistributed(MaxPooling1D(pool_size=3)))

model.add(TimeDistributed(Flatten()))

#model.add(TimeDistributed(Dense(108, activation='relu')))

#model.add(LSTM(60, return_sequences=True, input_shape=(108,1)))

model.add(LSTM(80, return_sequences=True))

#model.add(LSTM(40, return_sequences=True))

model.add(LSTM(20))

model.add(Dropout(0.5))

model.add(Dense(20, activation='relu'))

model.add(Dense(n_outputs, activation='sigmoid'))



for layer in model.layers:

    print(layer.output_shape)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])
epochs=15

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
model.save_weights('mnt8_2.hdf5')

model.save('mnt8_2.h5')
def divide_test_data(start,end):

    test_metadata = pd.read_csv('../input/metadata_test.csv') 

    test_index_list = test_metadata['signal_id'][start:end].values

    test_index_lists = []

    for i in range(len(test_index_list)):

        test_index_lists.append(str(test_index_list[i]))

    return test_index_lists

    test_index_lists_master.append(test_index_lists)

    y_pred = []

result=[]

flag=0

for i in range(len(test_index_lists_master)):

    test = np.array(pq.read_pandas(('../input/test.parquet'),columns=test_index_lists_master[i]).to_pandas().T)    

    print(len(test),len(test[0]))

    

    x_test = list(map(lambda x: feature_extractor(x), test))



    X_test = np.array(x_test).reshape(-1,x_test[0].shape[0])

    #X_test = np.expand_dims(X_test, axis=2)

    print(len(X_test),len(X_test[0]))

    n_signals = 1 #So far each instance is one signal. We will diversify them in next step

    n_outputs = 1 #Binary Classification





    n_steps, n_length = 250, 10



    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_signals))

    preds = model.predict(X_test)

    threshpreds = (preds>0.5)*1

    #print(len(threshpreds))

    #print(len(threshpreds[0]))

    #print(type(threshpreds))

    if flag ==0:

        result=threshpreds

        flag=1

    else:

        result=np.concatenate((result, threshpreds), axis=0)

    print(len(result))

    #print(len(result[0]))

    #print(type(result))

    gc.collect()   

    

sub = pd.read_csv('../input/sample_submission.csv')

print(len(sub))

print(len(result))

sub.target = result

sub.to_csv('sub16.csv',index=False)