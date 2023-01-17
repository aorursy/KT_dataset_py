# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
import multiprocessing

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras.backend as K #to define custom loss function
import tensorflow as tf #We'll use tensorflow backend here
import dask.dataframe as dd
import matplotlib.pyplot as plt

from tqdm import tnrange, tqdm_notebook
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, AveragePooling1D, Reshape, DepthwiseConv2D, SeparableConv2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

print(os.listdir("../input/"))
df = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')
df.head(5)
def dmdtize_single_band(df, dm_bins, dt_bins, col):
    n_points = df.shape[0]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins)), dtype='int')
    for i in range(n_points):
        for j in range(i+1, n_points):
            dmi = df.iloc[i][col]
            dmj = df.iloc[j][col]
            dti = df.iloc[i]['mjd']
            dtj = df.iloc[j]['mjd']
            
            dm = dmj - dmi if dtj > dti else dmi - dmj
            dt = abs(dtj - dti)
            
            dm_idx = min(np.searchsorted(dm_bins, dm), len(dm_bins)-1)
            dt_idx = min(np.searchsorted(dt_bins, dt), len(dt_bins)-1)
            
            dmdt_img[dm_idx, dt_idx] += 1
    return dmdt_img
def dmdtize_single_object(args):
    (df, object_id, base_dir) = args
    key = '{}/{}_dmdt.pkl'.format(base_dir, object_id)
    if os.path.isfile(key):
        return
    num_bands = 6
    dm_bins = [-8, -5, -3, -2.5, -2, -1.5, -1, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
    dt_bins = [1/145, 2/145, 3/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 240, 600, 960, 2000, 4000]
    dmdt_img = np.zeros((len(dm_bins), len(dt_bins), num_bands), dtype='int')
    
    mms = MinMaxScaler(feature_range=(-8, 8))
    df['local_tr_flux'] = mms.fit_transform(df['flux'].values.reshape(-1,1))
    
    max_points = 0
    for band_idx in range(num_bands):
        df_band = df.loc[df['passband'] == band_idx]
        dmdt_img[:, :, band_idx] = dmdtize_single_band(df_band, dm_bins, dt_bins, 'local_tr_flux')
        if band_idx == 0 or df_band.shape[0] > max_points:
            max_points = df_band.shape[0] #store max points to scale the image later
    
    max_pairs = (max_points*(max_points-1))//2
    dmdt_img = np.floor(255*dmdt_img/max_pairs + 0.99999).astype('int')
    with open(key, 'wb') as f:
        pickle.dump(dmdt_img, f)        
def dmdtize(df, base_dir='train'):
    objects = df['object_id'].drop_duplicates().values
    nobjects = len(objects)
    dmdt_img_dict = {}
    pool = multiprocessing.Pool()
    df_args = []
    for obj in objects:
        df_obj = df.loc[df['object_id'] == obj]
        df_args.append((df_obj, obj, base_dir))
    pool.map(dmdtize_single_object, df_args)
    pool.terminate()
objects = df['object_id'].drop_duplicates().values
def load_dmdt_images(objects, base_dir='train'):
    dmdt_img_dict = OrderedDict()
    for obj in objects:
        key = '{}/{}_dmdt.pkl'.format(base_dir, obj)
        if os.path.isfile(key):
            with(open(key, 'rb')) as f:
                dmdt_img_dict[obj] = pickle.load(f)
    return dmdt_img_dict
dmdt_img_dict = load_dmdt_images(objects, '../input/plasticc_dmdt_images/train/data1/plasticc/input/train')
X = np.array(list(dmdt_img_dict.values()), dtype='int')

df_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
labels = pd.get_dummies(df_meta.loc[df_meta['object_id'].isin(dmdt_img_dict.keys()) , 'target'])

y = labels.values
df_meta
labels
#TODO split X and y into train/test set. (Maybe a val set ?)
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
splits = list(splitter.split(X, y))[0]
train_ind, test_ind = splits
X_train = X[train_ind]
X_test  = X[test_ind]

y_train = y[train_ind]
y_test  = y[test_ind]
print(y_train.shape, y_test.shape)
def build_model(n_dm_bins=23, n_dt_bins=24, n_passbands=6, n_classes=14):
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=2, strides=1, depth_multiplier=8,
                     padding="valid", activation="elu",
                     input_shape=(n_dm_bins, n_dt_bins, n_passbands)))
    model.add(Dropout(0.1))
    model.add(SeparableConv2D(48, kernel_size=2, strides=1, depth_multiplier=2,
                    padding="valid", activation="elu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(32, activation="elu"))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes, activation="softmax"))
    print(model.summary())
    return model
n_classes=14
#assumes weights to be all ones as actual weights are hidden
#UPDATE - settings weights for classes 15 (idx=1) and 64(idx=7) to 2 based on LB probing post 
#https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194#397153
weights = np.ones(n_classes, dtype='float32') 
weights[1], weights[7] = 2, 2
epsilon = 1e-7
#number of objects per class
class_counts = df_meta.groupby('target')['object_id'].count().values 
#proportion of objects per class
class_proportions = class_counts/np.max(class_counts)
#set backend to float 32
K.set_floatx('float32')
#weighted multi-class log loss
def weighted_mc_log_loss(y_true, y_pred):
    y_pred_clipped = K.clip(y_pred, epsilon, 1-epsilon)
    #true labels weighted by weights and percent elements per class
    y_true_weighted = (y_true * weights)/class_proportions
    #multiply tensors element-wise and then sum
    loss_num = (y_true_weighted * K.log(y_pred_clipped))
    loss = -1*K.sum(loss_num)/K.sum(weights)
    
    return loss
y_true = K.variable(np.eye(14, dtype='float32'))
y_pred = K.variable(np.eye(14, dtype='float32'))
res = weighted_mc_log_loss(y_true, y_pred)
K.eval(res)
model = build_model()
model.compile(loss=weighted_mc_log_loss, optimizer=Adam(lr=0.002), metrics=['accuracy'])
checkPoint = ModelCheckpoint("./keras.model",monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)
class ReduceLRWithEarlyStopping(ReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        super(ReduceLRWithEarlyStopping, self).__init__(*args, **kwargs)
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        super(ReduceLRWithEarlyStopping, self).on_epoch_end(epoch, logs)
        old_lr = float(K.get_value(self.model.optimizer.lr))
        if self.wait >= self.patience and old_lr <= self.min_lr:
            # Stop training early
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        super(ReduceLRWithEarlyStopping, self).on_epoch_end(logs)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
reduce_lr = ReduceLRWithEarlyStopping(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=[X_test, y_test],shuffle=True,verbose=1,callbacks=[checkPoint, reduce_lr])
def plot_loss_acc(history):
    plt.plot(history['loss'][1:])
    plt.plot(history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history['acc'][1:])
    plt.plot(history['val_acc'][1:])
    plt.title('model Accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    plt.legend(['train','Validation'], loc='upper left')
    plt.show()
plot_loss_acc(history.history)
loss_acc = model.evaluate(X_test, y_test, batch_size=32)
print(loss_acc)
y_pred_test = model.predict(X_test)
classes = np.sort(df_meta['target'].drop_duplicates())
df_meta_test = df_meta.iloc[test_ind]
df_meta_test['pred_label'] = classes[np.argmax(y_pred_test, axis=1)]
df_meta_test.loc[df_meta_test.target == 15]
df_meta_test.loc[(df_meta_test.target == df_meta_test.pred_label) & (df_meta_test.target != 16) & (df_meta_test.target != 92) & (df_meta_test.target != 88)]
time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
save_file = 'model_{}.h5'.format(time_stamp)
model.save(save_file)
