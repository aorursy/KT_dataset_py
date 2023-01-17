from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)

import pandas as pd

import pyarrow.parquet as pq

from tqdm import trange,tqdm

import os

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import keras

import keras.backend as K

from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten,GlobalAveragePooling1D,AveragePooling1D,GlobalMaxPooling1D,BatchNormalization,Activation,Bidirectional

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Sequential

import tensorflow as tf

import warnings

warnings.filterwarnings("ignore")

from scipy import signal

import gc
compression_bucket_size = 300
# Load Data

def load_data(parquet_data, csv_metadata):

    pq_data = np.array(pq.read_pandas(('../input/'+parquet_data)).to_pandas().T)

    metadata = pd.read_csv('../input/'+csv_metadata) 

    target = metadata['target'][:len(pq_data)].values

    return pq_data,target



from numpy.fft import *

def filter_signal(signal, threshold=1e8):

    fourier = rfft(signal)

    frequencies = rfftfreq(signal.size, d=20e-3/signal.size)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)



# Subtract de-noised data from the raw signal data to process, or normalize(?), data. 

def signal_processing(data):

    return abs(np.round((data-filter_signal(data,threshold=1e3)),2))





def train_validate_split(data,data_target,validate_size):

    metadata = pd.read_csv('../input/metadata_train.csv') 

    signal_id_1 = list(metadata[metadata['target']==1]['signal_id'])

    signal_id_0 = list(metadata[metadata['target']==0]['signal_id'])

    train_1 = signal_id_1[0:int(len(signal_id_1)*(1-validate_size))]

    validate_1 = signal_id_1[int(len(signal_id_1)*(1-validate_size)):]

    train_0 = signal_id_0[0:int(len(signal_id_0)*(1-validate_size))]

    validate_0 = signal_id_0[int(len(signal_id_0)*(1-validate_size)):]

    

    data_train = data[sorted(np.concatenate((train_0,train_1)))]

    data_train_target = data_target[sorted(np.concatenate((train_0,train_1)))]

    data_validate = data[sorted(np.concatenate((validate_0,validate_1)))]

    data_validate_target = data_target[sorted(np.concatenate((validate_0,validate_1)))]  

    

    return data_train, data_train_target, data_validate, data_validate_target



# Reduce sample size from 800000 to 800000/bucket_size while not losing information by extracting features : std, mean, max, min

def compress_data_and_extract_features(data,bucket_size):

    data_bucket_std, data_bucket_mean, data_bucket_percentile_0, data_bucket_percentile_1, data_bucket_percentile_25, data_bucket_percentile_50, data_bucket_percentile_75, data_bucket_percentile_99, data_bucket_percentile_100 = [],[],[],[],[],[],[],[],[]

    

        

    for i in trange(data.shape[0]):

        holder_std, holder_mean, holder_percentile,holder_0,holder_1,holder_25,holder_50,holder_75,holder_99,holder_100  = [],[],[],[],[],[],[],[],[],[]

        #percentile_threshhold = np.percentile(abs(data[i]),99.97)

        for j in range(0,data.shape[1],bucket_size):

            holder_std.append(abs(data[i][j:(j+bucket_size)]).std())

            holder_mean.append(abs(data[i][j:(j+bucket_size)]).mean())

            holder_percentile=np.percentile(abs(data[i][j:(j+bucket_size)]),[0, 1, 25, 50, 75, 99, 100])

            holder_0.append(holder_percentile[0])

            holder_1.append(holder_percentile[1])

            holder_25.append(holder_percentile[2])

            holder_50.append(holder_percentile[3])

            holder_75.append(holder_percentile[4])

            holder_99.append(holder_percentile[5])

            holder_100.append(holder_percentile[6])

            #holder_peaks.append(sum(abs(data[i][j:(j+bucket_size)])>percentile_threshhold))           

            

        data_bucket_std.append(holder_std)

        data_bucket_mean.append(holder_mean)

        data_bucket_percentile_0.append(holder_0)

        data_bucket_percentile_1.append(holder_1)

        data_bucket_percentile_25.append(holder_25)

        data_bucket_percentile_50.append(holder_50)

        data_bucket_percentile_75.append(holder_75)

        data_bucket_percentile_99.append(holder_99)

        data_bucket_percentile_100.append(holder_100)

        #data_bucket_peaks.append(holder_peaks)        

    return np.asarray(data_bucket_std), np.asarray(data_bucket_mean), np.asarray(data_bucket_percentile_0), np.asarray(data_bucket_percentile_1), np.asarray(data_bucket_percentile_25), np.asarray(data_bucket_percentile_50), np.asarray(data_bucket_percentile_75), np.asarray(data_bucket_percentile_99), np.asarray(data_bucket_percentile_100)



# Reshape Input Data of multiple features into single input for LSTM Input

def LSTM_reshape_dstack(combined_data_list):      

    for i in range(len(combined_data_list)):

        combined_data_list[i]=combined_data_list[i].reshape(combined_data_list[i].shape[0],combined_data_list[i].shape[1],1)        

    return np.dstack(combined_data_list)
# singal = 6

# #  3,    4,    5,  201,  202,  228,  229,  230,  270,  271



# plt.figure(figsize=(10,3))

# t = np.arange(0,len(train[0]))

# plt.plot(t,train[singal])

# plt.show()
train,target_train = load_data('train.parquet','metadata_train.csv')

for i in trange(len(train)):  

    train[i]=signal_processing(train[i])

#     train_min = min(train[i])

#     train_max = max(train[i])

#     train[i]= (train[i]-train_min)/(train_max-train_min)*2-1

data_train, data_train_target, data_validate, data_validate_target = train_validate_split(train,target_train,0.3)

del train,target_train

gc.collect()

#train_std,train_mean,train_max,train_25,train_50,train_75,train_peaks = compress_data_and_extract_features(data_train,compression_bucket_size)

train_std,train_mean,train_0,train_1,train_25,train_50,train_75,train_99,train_100 = compress_data_and_extract_features(data_train,compression_bucket_size)

#train_LSTM = LSTM_reshape_dstack([train_std,train_mean,train_max,train_25,train_50,train_75,train_peaks])

train_LSTM = LSTM_reshape_dstack([train_std,train_mean,train_0,train_1,train_25,train_50,train_75,train_99,train_100])

train_LSTM_backup = train_LSTM.copy()

train_LSTM = train_LSTM.reshape(train_LSTM.shape[0],1,train_LSTM.shape[1],train_LSTM.shape[2])



del train_std,train_mean,train_0,train_1,train_25,train_50,train_75,train_99,train_100#,train_peaks

gc.collect()



#validate_std,validate_mean,validate_max,validate_25,validate_50,validate_75,validate_peaks = compress_data_and_extract_features(data_validate,compression_bucket_size)

validate_std,validate_mean,validate_0,validate_1,validate_25,validate_50,validate_75,validate_99,validate_100 = compress_data_and_extract_features(data_validate,compression_bucket_size)

#validate_LSTM = LSTM_reshape_dstack([validate_std,validate_mean,validate_0,validate_1,validate_25,validate_50,validate_75,validate_99,validate_100])

validate_LSTM = LSTM_reshape_dstack([validate_std,validate_mean,validate_0,validate_1,validate_25,validate_50,validate_75,validate_99,validate_100])

validate_LSTM_backup = validate_LSTM.copy()

validate_LSTM = validate_LSTM.reshape(validate_LSTM.shape[0],1,validate_LSTM.shape[1],validate_LSTM.shape[2])



#del validate_std,validate_mean,validate_max,validate_25,validate_50,validate_75,validate_peaks

del validate_std,validate_mean,validate_0,validate_1,validate_25,validate_50,validate_75,validate_99,validate_100

gc.collect()



# For easier readability

num_signals = train_LSTM.shape[0]

num_timesteps = train_LSTM.shape[2]

num_features = train_LSTM.shape[3]
def keras_auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc

def matthews_correlation(y_true, y_pred):

    '''Calculates the Matthews correlation coefficient measure for quality

    of binary classification problems.

    '''

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos



    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos



    tp = K.sum(y_pos * y_pred_pos)

    tn = K.sum(y_neg * y_pred_neg)



    

    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg)



    numerator = (tp * tn - fp * fn)

    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))



    return numerator/(denominator+K.epsilon())
#num_signals, num_timesteps, num_features = train_LSTM.shape[2]



model = Sequential()

# num_timesteps = 800000

# num_features = 6



# Define CNN Model

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=6), input_shape=(None,num_timesteps,num_features)))

model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=6)))

model.add(TimeDistributed(Activation('relu')))

model.add(TimeDistributed(GlobalMaxPooling1D()))





model.add(TimeDistributed(Flatten()))

# Define LSTM Model

model.add(LSTM(128))

model.add(Dropout(0.5))

model.add(Dense(16, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

callbacks = [EarlyStopping(monitor='val_matthews_correlation', patience=20),

            ModelCheckpoint(filepath='best_model.h5', monitor='val_matthews_correlation', save_best_only=True,mode='max')]

#callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_matthews_correlation', save_best_only=True,mode='max')]


model.fit(train_LSTM, data_train_target, validation_data=(validate_LSTM[0:1308],data_validate_target[0:1308]),epochs=200, batch_size=128, verbose=1,callbacks=callbacks)


model.fit(train_LSTM, data_train_target, validation_data=(validate_LSTM[1308:],data_validate_target[1308:]),epochs=200, batch_size=128, verbose=1,callbacks=callbacks)
callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='matthews_correlation', save_best_only=True,mode='max')]



model.fit(np.concatenate((train_LSTM,validate_LSTM)), np.concatenate((data_train_target,data_validate_target)),epochs=10, batch_size=128, verbose=1,callbacks=callbacks)
# #num_signals, num_timesteps, num_features = train_LSTM.shape[2]



# model = Sequential()

# # num_timesteps = 800000

# # num_features = 6



# # Define CNN Model

# model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2), input_shape=(None,num_timesteps,num_features)))

# model.add(TimeDistributed(Activation('relu')))

# model.add(TimeDistributed(MaxPooling1D(pool_size=6)))





# model.add(TimeDistributed(Conv1D(filters=16, kernel_size=2)))

# model.add(TimeDistributed(Activation('relu')))

# model.add(TimeDistributed(MaxPooling1D(pool_size=6)))



# model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2)))

# model.add(TimeDistributed(Activation('relu')))

# model.add(TimeDistributed(MaxPooling1D(pool_size=6)))



# model.add(TimeDistributed(Conv1D(filters=4, kernel_size=2)))

# model.add(TimeDistributed(Activation('relu')))

# model.add(TimeDistributed(MaxPooling1D(pool_size=6)))







# # model.add(TimeDistributed(GlobalMaxPooling1D()))



# # model.add(TimeDistributed(Conv1D(filters=16, kernel_size=4)))

# # model.add(TimeDistributed(Activation('relu')))

# # #model.add(TimeDistributed(BatchNormalization()))

# # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

# # model.add(Dropout(0.2))



# # model.add(TimeDistributed(Conv1D(filters=16, kernel_size=2)))

# # model.add(TimeDistributed(Activation('relu')))

# # #model.add(TimeDistributed(BatchNormalization()))

# # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

# # model.add(Dropout(0.2))







# model.add(TimeDistributed(Flatten()))

# # Define LSTM Model

# model.add(LSTM(128))

# model.add(Dropout(0.5))

# model.add(Dense(32, activation='relu'))

# model.add(Dropout(0.5))

# model.add(Dense(1, activation='sigmoid'))

# model.summary()



# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])



# #callbacks = [EarlyStopping(monitor='val_matthews_correlation', patience=50),

# #             ModelCheckpoint(filepath='best_model.h5', monitor='val_matthews_correlation', save_best_only=True,mode='max')]

# callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_matthews_correlation', save_best_only=True,mode='max')]

# model.fit(train_LSTM, data_train_target, validation_data=(validate_LSTM,data_validate_target),epochs=300, batch_size=64, verbose=1,callbacks=callbacks)

del train_LSTM, data_train_target, validate_LSTM, data_validate_target

gc.collect()



# model.save_weights('model1.hdf5')

model.load_weights('best_model.h5')
def divide_test_data(start,end):

    test_metadata = pd.read_csv('../input/metadata_test.csv') 

    test_index_list = test_metadata['signal_id'][start:end].values

    test_index_lists = []

    for i in range(len(test_index_list)):

        test_index_lists.append(str(test_index_list[i]))

    return test_index_lists

    test_index_lists_master.append(test_index_lists)

    

# Process test data in 6 chunks due to memory issues

test_index_lists_master=[]

test_index_lists_master.append(divide_test_data(0,3390))

test_index_lists_master.append(divide_test_data(3390,6780))

test_index_lists_master.append(divide_test_data(6780,10170))

test_index_lists_master.append(divide_test_data(10170,13560))

test_index_lists_master.append(divide_test_data(13560,16950))

test_index_lists_master.append(divide_test_data(16950,20337))
y_pred = []

for i in range(len(test_index_lists_master)):

    test = np.array(pq.read_pandas(('../input/test.parquet'),columns=test_index_lists_master[i]).to_pandas().T)    

    for i in range(len(test)):

        test[i]=signal_processing(test[i])    

#         test_min = min(test[i])

#         test_max = max(test[i])

#         test[i]= (test[i]-test_min)/(test_max-test_min)*2-1    

    

    

    #test_std,test_mean,test_max,test_25,test_50,test_75,test_peaks = compress_data_and_extract_features(test,compression_bucket_size)

    test_std,test_mean,test_0,test_1,test_25,test_50,test_75,test_99,test_100 = compress_data_and_extract_features(test,compression_bucket_size)

    del test

    #test_LSTM = LSTM_reshape_dstack([test_std,test_mean,test_max,test_25,test_50,test_75,test_peaks])

    test_LSTM = LSTM_reshape_dstack([test_std,test_mean,test_0,test_1,test_25,test_50,test_75,test_99,test_100])

    #del test_std,test_mean,test_max,test_25,test_50,test_75,test_peaks

    del test_std,test_mean,test_0,test_1,test_25,test_50,test_75,test_99,test_100

    test_LSTM = test_LSTM.reshape(test_LSTM.shape[0],1,test_LSTM.shape[1],test_LSTM.shape[2])

    y_pred.append(model.predict(test_LSTM))

    del test_LSTM

    gc.collect()   
y_pred_final=np.concatenate((y_pred[0],y_pred[1],y_pred[2],y_pred[3],y_pred[4],y_pred[5]))

# Do this temporarily since target and number of signals didn't match.

#y_pred_final=np.append(y_pred_final,0)

test_metadata = pd.read_csv('../input/metadata_test.csv') 

test_metadata['target']=y_pred_final

test_metadata=test_metadata.drop(columns=['phase','id_measurement'])
threshhold=0.5

test_metadata['target'][test_metadata['target']>=threshhold]=1

test_metadata['target'][test_metadata['target']<threshhold]=0
test_metadata['target']=test_metadata.target.astype(int)
test_metadata.to_csv('submission.csv',index=False)
print(len(y_pred[0]))

print(sum(y_pred[0]))



print(len(y_pred[1]))

print(sum(y_pred[1]))



print(len(y_pred[2]))

print(sum(y_pred[2]))



print(len(y_pred[3]))

print(sum(y_pred[3]))



print(len(y_pred[4]))

print(sum(y_pred[4]))



print(len(y_pred[5]))

print(sum(y_pred[5]))


