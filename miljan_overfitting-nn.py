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
import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Input

from keras.layers import Dropout, BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.optimizers import Adam

from sklearn.model_selection import KFold



import gc

import os







from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold



import tensorflow as tf

from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Concatenate, BatchNormalization, GlobalAveragePooling1D, LeakyReLU

from keras.models import Model, Sequential

from keras import regularizers
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
# Basic feature engineering



idx = features = train_df.columns.values[2:202]

for df in [test_df, train_df]:

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)
# Implementation found on kaggle 

def rank_gauss(x):

    from scipy.special import erfinv

    N = x.shape[0]

    temp = x.argsort()

    rank_x = temp.argsort() / N

    rank_x -= rank_x.mean()

    rank_x *= 2

    efi_x = erfinv(rank_x)

    efi_x -= efi_x.mean()

    return efi_x
X_train = train_df.drop(columns=['ID_code', 'target'])

y_train = train_df['target'].values

X_test = test_df.drop(columns=['ID_code'])
for i in X_train.columns:

    #print('Categorical: ',i)

    X_train[i] = rank_gauss(X_train[i].values)
for i in X_test.columns:

    #print('Categorical: ',i)

    X_test[i] = rank_gauss(X_test[i].values)
X_train = X_train.values

X_test = X_test.values
from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback

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
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import BatchNormalization

from keras.layers import Flatten

from keras.layers import GRU, GaussianDropout,ThresholdedReLU, ReLU, Activation

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam, Adamax

from keras.activations import selu

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
def model_creator():

  nn = Sequential()

  nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = 208))

  nn.add(PReLU())

  nn.add(Dropout(.3))

  nn.add(Dense(units = 160 , kernel_initializer = 'normal'))

  nn.add(PReLU())

  nn.add(BatchNormalization())

  nn.add(Dropout(.3))

  nn.add(Dense(units = 64 , kernel_initializer = 'normal'))

  nn.add(PReLU())

  nn.add(BatchNormalization())

  nn.add(GaussianDropout(.2))

  nn.add(Dense(units = 26, kernel_initializer = 'normal'))

  nn.add(PReLU())

  nn.add(BatchNormalization())

  nn.add(GaussianDropout(.2))

  nn.add(Dense(units = 12, kernel_initializer = 'normal'))

  nn.add(PReLU())

  nn.add(BatchNormalization())

  nn.add(GaussianDropout(.2))

  nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))



  return nn
folds = KFold(n_splits=10, shuffle=True, random_state=42)

sub_preds = np.zeros(X_test.shape[0])



for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):

    trn_x, trn_y = X_train[trn_idx], y_train[trn_idx]

    val_x, val_y = X_train[val_idx], y_train[val_idx]

    

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=2, min_lr=0.0000001, mode='min')

    h5_path = "model.h5"

    checkpoint = ModelCheckpoint(h5_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')



    early_stopping = EarlyStopping(monitor='val_loss', patience=4)

    

    print( 'Setting up neural network...' )



    nn = model_creator()

    nn.compile(loss='binary_crossentropy', optimizer='adam')

    

    print( 'Fitting neural network...' )

    nn.fit(trn_x, trn_y, validation_data = (val_x, val_y), epochs=100, verbose=2,

          callbacks=[reduce_lr, checkpoint, early_stopping, roc_callback(training_data=(trn_x, trn_y),validation_data=(val_x, val_y))])

    nn.load_weights(h5_path)

    print( 'Predicting...' )

    sub_preds += nn.predict(X_test).flatten().clip(0,1) / folds.n_splits

    

    gc.collect()
sub = pd.DataFrame()

sub["ID_code"] = test_df["ID_code"]

sub["target"] = sub_preds

sub.to_csv("submission.csv", index=False)