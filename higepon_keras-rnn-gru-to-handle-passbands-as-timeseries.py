# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import time
import tensorflow as tf
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import GRU, Dense, Activation, Dropout, CuDNNGRU, concatenate, Input
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/training_set.csv')
train.head()
ss = StandardScaler()
train[['mjd', 'flux', 'flux_err']] = ss.fit_transform(train[['mjd', 'flux', 'flux_err']])
train.head()
def get_timeseries_by_column_name(train, passband, column_name):
    train_passband = train[train.passband == passband]
    ## This is where magic happens
    train_column_timeseries = train_passband.groupby(['object_id', 'passband'])[column_name].apply(lambda df: df.reset_index(drop=True)).unstack()
    train_column_timeseries.fillna(0, inplace=True)
    train_column_timeseries  = train_column_timeseries.reset_index()
    train_column_timeseries['feature_id'] = column_name
    return train_column_timeseries

def get_timeseries(train, passband):
    df = pd.concat([get_timeseries_by_column_name(train, passband, column_name) for column_name in ['mjd', 'flux', 'flux_err', 'detected']])
    df = df.sort_values(['object_id', 'passband'])
    return df
# One example for flux where passband=0
get_timeseries_by_column_name(train, passband=0, column_name='flux').head()
# for each passband we get input rows for mjd, flux, flux_err, detected
train_list = []
for passband in range(0, 6):
    train_passband =  get_timeseries(train, passband=passband)
    train_list.append(train_passband)
    print("train_list[%d]" % passband, train_list[passband].shape)
num_features = len(['mjd', 'flux', 'flux_err', 'detected'])
drop_features= [ "feature_id", "object_id", "passband"]
X_train_list = []
for passband in range(0, 6):
    num_columns = len(train_list[passband].columns) - len(drop_features)
    print("num_columns[%d]:" % passband, num_columns)
    X_train_list.append(train_list[passband].drop(drop_features, axis=1).values.reshape(-1, num_features, num_columns).transpose(0, 2, 1))
    print("X_train_list[%d].shape:" % passband, X_train_list[passband].shape)
meta_train = pd.read_csv('../input/training_set_metadata.csv')
meta_train.head()
# List all classes
classes = sorted(meta_train.target.unique())
classes
class_map = dict()
for i,val in enumerate(classes):
    class_map[val] = i
class_map
# We only need target for earch object_id, so using feature_id=='flux'
merged_train = train_list[0][train_list[0].feature_id == 'flux'].merge(meta_train, on='object_id', how='left')
merged_train = merged_train.drop(['ra',	'decl',	'gal_l',	'gal_b',	'ddf',	'hostgal_specz',	'hostgal_photoz',	'hostgal_photoz_err',	'distmod',	'mwebv'], axis=1)
merged_train.head()
targets = merged_train.target
target_map = np.zeros((targets.shape[0],))
target_map = np.array([class_map[val] for val in targets])
Y = to_categorical(target_map)
Y.shape
def multi_weighted_logloss(y_ohe, y_p):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1-1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set 
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos    
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss

# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69795
def mywloss(y_true,y_pred):  
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss

def plot_loss_acc(history):
    plt.plot(history.history['loss'][1:])
    plt.plot(history.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['acc'][1:])
    plt.plot(history.history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()

n_classes = len(classes)

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def build_model():
    def basic_layer(input_):
        output = GRU(64,
                     kernel_initializer=weight_variable,
                     dropout=0.5,
                     recurrent_dropout=0.5,
                     return_sequences=True)(input_)
        output = Dropout(0.5)(output)
        output = GRU(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(output)
        output = GRU(16, dropout=0.5, recurrent_dropout=0.5)(output)
        output = Dense(32)(output)
        return output
    
    # Keras functional API supports multiple inputs!
    input0 = Input(shape=(None, num_features), dtype='float32', name='passband0')
    input1 = Input(shape=(None, num_features), dtype='float32', name='passband1')
    input2 = Input(shape=(None, num_features), dtype='float32', name='passband2')
    input3 = Input(shape=(None, num_features), dtype='float32', name='passband3')
    input4 = Input(shape=(None, num_features), dtype='float32', name='passband4')
    input5 = Input(shape=(None, num_features), dtype='float32', name='passband5')
    
    merged_output = concatenate([basic_layer(input0),basic_layer(input1),basic_layer(input2),basic_layer(input3),basic_layer(input4),basic_layer(input5)])
    merged_output = Dense(64)(merged_output)
    final_output = Dense(len(classes), activation='softmax')(merged_output)
    return Model(inputs=[input0, input1, input2, input3, input4, input5], outputs=[final_output])
y_count = Counter(target_map)
wtable = np.zeros((len(classes),))
for i in range(len(classes)):
    wtable[i] = y_count[i] / target_map.shape[0]
batch_size = 512
y_map = target_map
y_categorical = Y
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
start = time.time()
clfs = []
oof_preds = np.zeros((len(X_train_list[0]), len(classes)))
epochs = 10 #5000

for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
    checkPoint = ModelCheckpoint('./keras.model',monitor='val_loss',mode = 'min', save_best_only=True, verbose=0)
    X0 = X_train_list[0]
    X1 = X_train_list[1]
    X2 = X_train_list[2]
    X3 = X_train_list[3]
    X4 = X_train_list[4]
    X5 = X_train_list[5]
    x_train0, x_train1, x_train2, x_train3, x_train4, x_train5, y_train = X0[trn_], X1[trn_], X2[trn_], X3[trn_], X4[trn_], X5[trn_], y_categorical[trn_]
    x_valid0, x_valid1, x_valid2, x_valid3, x_valid4, x_valid5, y_valid = X0[val_], X1[val_], X2[val_], X3[val_], X4[val_], X5[val_], y_categorical[val_]
    
    model = build_model()    
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)#
    early_stopping = EarlyStopping(monitor='loss', patience=30, verbose=1)
    model.compile(loss=mywloss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit([x_train0, x_train1, x_train2, x_train3, x_train4, x_train5], y_train,
                    validation_data=[[x_valid0, x_valid1, x_valid2, x_valid3, x_valid4, x_valid5], y_valid], 
                    epochs=epochs,
                        batch_size=batch_size,
                    shuffle=True,verbose=1,callbacks=[checkPoint, early_stopping])      
    plot_loss_acc(history)
    
    print('Loading Best Model')
    model.load_weights('./keras.model')
    # # Get predicted probabilities for each class
    oof_preds[val_, :] = model.predict([x_valid0, x_valid1, x_valid2, x_valid3, x_valid4, x_valid5],batch_size=batch_size)
    print(multi_weighted_logloss(y_valid, model.predict([x_valid0, x_valid1, x_valid2, x_valid3, x_valid4, x_valid5],batch_size=batch_size)))
    clfs.append(model)
    
print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_categorical,oof_preds))

elapsed_time = time.time() - start
print("elapsed_time:", elapsed_time)