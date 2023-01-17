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
#import neccessary libraries

from keras.layers import LSTM,GRU,Dense,Bidirectional,Dropout

from keras.callbacks import *

from keras import layers

from keras.models import Sequential

from keras.optimizers import RMSprop,Adam

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback

from matplotlib import pyplot as plt
#import data

PATH="../input/"

data_train=pd.read_csv(f'{PATH}train.csv')

data_test=pd.read_csv(f'{PATH}test.csv')
#see the shapes of data

data_train.shape,data_test.shape
features=list(data_train.columns.values[2:])

target=['target']
#normalizing data for nn

scaler=StandardScaler()

data_train_scaled=data_train.copy()

data_test_scaled=data_test.copy()

data_train_scaled[features]=scaler.fit_transform(data_train[features].T).T

data_test_scaled[features]=scaler.transform(data_test[features].T).T
#standard split on train and validation

X_train,X_valid,y_train,y_valid=train_test_split(data_train_scaled[features],

                                              data_train_scaled[target])
#create testset and check sizes

X_test=data_test_scaled[features]

X_train.shape,X_valid.shape,X_test.shape
#create custom auc metrics

class roc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]





    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)

        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
#create callback procedures

earlystopper = EarlyStopping(patience=8, verbose=1)

checkpointer = ModelCheckpoint(filepath = 'model_tranz.hdf5',

                               verbose=1,

                               save_best_only=True, save_weights_only = True)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=2, min_lr=0.000001, verbose=1,cooldown=1)
#create simple NN model

model = Sequential()

model.add(Dense(512,activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])
#train the model

history=model.fit(X_train.values,y_train.values,epochs=50,batch_size=2048,

                 validation_data=(X_valid,y_valid.values),

                 callbacks=[roc_callback(training_data=(

                     X_train.values, y_train.values),

                                         validation_data=(X_valid.values, y_valid.values)),

                           earlystopper, checkpointer, reduce_lr])
#0.85 for simple nn... not so bad

#next step is adding one more layer 

model = Sequential()

model.add(Dense(32,activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32,activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])
history_2=model.fit(X_train.values,y_train.values,epochs=50,batch_size=2048,

                 validation_data=(X_valid,y_valid.values),

                 callbacks=[roc_callback(training_data=(

                     X_train.values, y_train.values),

                                         validation_data=(X_valid.values, y_valid.values)),

                           earlystopper, checkpointer, reduce_lr])
#then go to lstm

#first of all edit Xs

X_train_rnn=np.reshape(X_train.values,(X_train.shape[0],1,X_train.shape[1]))

X_valid_rnn=np.reshape(X_valid.values,(X_valid.shape[0],1,X_valid.shape[1]))

X_test_rnn=np.reshape(data_test_scaled[features].values,(data_test_scaled[features].shape[0],1,

                                                     data_test_scaled[features].shape[1]))

X_train_rnn.shape,X_valid_rnn.shape
#create simple lstm

model = Sequential()

model.add(LSTM(32,batch_size=2048,batch_input_shape=(None,1,X_train.shape[-1]),

              input_shape=(1,X_train_rnn.shape[-1])))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=1e-2),loss='binary_crossentropy',metrics=['accuracy'])

history_lstm=model.fit(X_train_rnn,y_train.values,epochs=50,batch_size=2048,

                 validation_data=(X_valid_rnn,y_valid.values),

                 callbacks=[roc_callback(training_data=(

                     X_train_rnn, y_train.values),

                                         validation_data=(X_valid_rnn, y_valid.values)),

                           earlystopper, checkpointer, reduce_lr])
#create simple gru

model = Sequential()

model.add(GRU(32,batch_size=2048,batch_input_shape=(None,1,X_train.shape[-1]),

              input_shape=(1,X_train.shape[-1])))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=1e-2),loss='binary_crossentropy',metrics=['accuracy'])

history_gru=model.fit(X_train_rnn,y_train.values,epochs=50,batch_size=2048,

                 validation_data=(X_valid_rnn,y_valid.values),

                 callbacks=[roc_callback(training_data=(

                     X_train_rnn, y_train.values),

                                         validation_data=(X_valid_rnn, y_valid.values)),

                           earlystopper, checkpointer, reduce_lr])
#create bidirectional gru

model = Sequential()

model.add(Bidirectional(GRU(32,batch_size=1024,

              input_shape=(1,X_train.shape[-1]))))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=1e-2),loss='binary_crossentropy',metrics=['accuracy'])

history_bidir=model.fit(X_train_rnn,y_train.values,epochs=50,batch_size=2048,

                 validation_data=(X_valid_rnn,y_valid.values),

                 callbacks=[roc_callback(training_data=(

                     X_train_rnn, y_train.values),

                                         validation_data=(X_valid_rnn, y_valid.values)),

                           earlystopper, checkpointer, reduce_lr])
#create stacked GRU

model = Sequential()

model.add(GRU(32,batch_size=2048,batch_input_shape=(None,1,X_train.shape[-1]),

              return_sequences=True,dropout=0.1,recurrent_dropout=0.5,

              input_shape=(1,X_train.shape[-1])))

model.add(GRU(64,batch_size=2048,batch_input_shape=(None,1,X_train.shape[-1]),

              return_sequences=False,dropout=0.1,recurrent_dropout=0.5,

              input_shape=(1,X_train.shape[-1])))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=1e-2),loss='binary_crossentropy',metrics=['accuracy'])

history_gru_stacked=model.fit(X_train_rnn,y_train.values,epochs=50,batch_size=2048,

                 validation_data=(X_valid_rnn,y_valid.values),

                 callbacks=[roc_callback(training_data=(

                     X_train_rnn, y_train.values),

                                         validation_data=(X_valid_rnn, y_valid.values)),

                           earlystopper, checkpointer, reduce_lr])
#final plot for presenting results

fig, ax = plt.subplots()

val_loss_simple_nn=history.history['val_loss']

val_loss_2layers_nn=history_2.history['val_loss']

val_loss_lstm=history_lstm.history['val_loss']

val_loss_gru=history_gru.history['val_loss']

val_loss_bidir=history_bidir.history['val_loss']

val_loss_gru2=history_gru_stacked.history['val_loss']



epochs_1=range(1,len(val_loss_simple_nn)+1)

epochs_2=range(1,len(val_loss_2layers_nn)+1)

epochs_3=range(1,len(val_loss_lstm)+1)

epochs_4=range(1,len(val_loss_gru)+1)

epochs_5=range(1,len(val_loss_bidir)+1)

epochs_6=range(1,len(val_loss_gru2)+1)



ax.plot(epochs_1,val_loss_simple_nn,'b',label='val_loss_simple_nn')

ax.plot(epochs_2,val_loss_2layers_nn,'r',label='val_loss_2layers_nn')

ax.plot(epochs_3,val_loss_lstm,'go',label='val_loss_lstm')

ax.plot(epochs_4,val_loss_gru,'yo',label='val_loss_gru')

ax.plot(epochs_5,val_loss_bidir,'bo',label='val_loss_bidir')

ax.plot(epochs_6,val_loss_gru2,'ro',label='val_loss_gru2layers')



plt.title('Losses on models')

plt.legend()

plt.show()