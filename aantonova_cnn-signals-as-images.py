import numpy as np
import pandas as pd
from scipy.stats import skew
from matplotlib import pylab as plt
from sklearn.model_selection import StratifiedKFold
import gc
import time
import warnings
warnings.simplefilter(action = 'ignore')
import tensorflow as tf
from tensorflow import keras
train_all = pd.read_csv('../input/training_set.csv')
train_all.head()
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train_meta.head()
galactic_objects = list(train_meta[np.isnan(train_meta['distmod'])]['object_id'])
len(galactic_objects)
galactic_set = train_all[train_all['object_id'].isin(galactic_objects)]
galactic_set.shape
galactic_target = train_meta[train_meta['object_id'].isin(galactic_objects)][['object_id', 'target']]
galactic_target.shape
galactic_classes = sorted(galactic_target['target'].unique())
galactic_classes
del train_all, train_meta
gc.collect()
# This function calculate some statistics for timeseries. 
# In this model I use only the median of the secont column for defining step size of binnarizing.
# Others statistics using in another models.

def get_stats(df):
    groups = df.groupby('passband')
    res = groups['mjd'].apply(np.count_nonzero).values
    res = np.vstack((res, groups['mjd'].apply(np.asarray).apply(lambda x: np.median(x[1:] - x[:-1])))) # This
    #res = np.vstack((res, groups['flux'].apply(np.mean)))
    #res = np.vstack((res, groups['flux'].apply(np.std)))
    #res = np.vstack((res, groups['flux'].apply(skew)))
    #res = np.vstack((res, groups['flux_err'].apply(np.mean)))
    #res = np.vstack((res, groups['flux_err'].apply(np.std)))
    #res = np.vstack((res, groups['flux_err'].apply(skew)))
    #res = np.vstack((res, groups['detected'].apply(np.mean)))
    #res = np.vstack((res, groups['detected'].apply(np.std)))
    
    return np.transpose(res)
# This function converts numpy array from source dataset into 3-channels binned array with fixed width.
# Columns in array must contain values: mjd, passband, flux, flux_err, detected.
# Used in Batch Generator wich adds zeros to the equal length.

def to_binned_timeseries(ndar, step):
    warnings.simplefilter(action = 'ignore')
    
    # the first time for object
    start = np.min(ndar[:, 0])
    # sequence duration for object
    mjd_lendth = np.max(ndar[:, 0]) - start
    # count of bins for object timeseries
    timeseries_lendth = int(mjd_lendth / step) + 1
    # matrix for counts in each bin for each row
    cnt = np.zeros((6, timeseries_lendth))
    # matrix for result with 3 channels: flux, flux_err, detected
    # corresponds to data_format = 'channels_last' for CPU
    res = np.zeros((6, timeseries_lendth, 3))
    
    # loop for rows in sourse array for calculating summs
    for i in range(ndar.shape[0]):
        row = ndar[i, :]
        col_num = int((row[0] - start) / step)
        cnt[int(row[1]), col_num] += 1
        res[int(row[1]), col_num, 0] += row[2]
        res[int(row[1]), col_num, 1] += row[3]
        res[int(row[1]), col_num, 2] += row[4]
        
    # get mean values exclude nans
    res[:, :, 0] /= cnt
    res[:, :, 1] /= cnt
    res[:, :, 2] /= cnt
    
    # normalizing flux channels by rows
    for channel in range(2):
        means = np.reshape([np.mean(res[i, ~np.isnan(res[i, :, channel]), channel]) for i in range(6)]*timeseries_lendth, 
                           (6, timeseries_lendth), order = 'F')
        stds = np.reshape([np.std(res[i, ~np.isnan(res[i, :, channel]), channel]) for i in range(6)]*timeseries_lendth, 
                          (6, timeseries_lendth), order = 'F')
        res[:, :, channel] = (res[:, :, channel] - means) / stds
        
    # replacing nans to zeros
    res = np.nan_to_num(res)
        
    return res
#Calculating of constant for this model and dataset

MAX_LENDTH = -1
for obj in galactic_objects:
    ndar = galactic_set[galactic_set['object_id'] == obj][['mjd', 'passband', 'flux', 'flux_err', 'detected']]
    stats = get_stats(ndar)
    data = to_binned_timeseries(ndar.values, np.median(stats[:, 1]))
    if data.shape[1] > MAX_LENDTH:
        MAX_LENDTH = data.shape[1]
        
print('Count of columns in `image`:', MAX_LENDTH)
class BatchGenerator(keras.utils.Sequence):
    
    def __init__(self, X, y, batch_size = 32, predict = False):
        self.X = X
        self.index = list(X['object_id'].unique())
        self.y = y
        self.batch_size = batch_size
        self.predict = predict

        if not predict:
            self.on_epoch_end()
        
    def __getitem__(self, index_batch):
        idx = self.index[index_batch * self.batch_size : (index_batch + 1) * self.batch_size]
        batch = np.zeros((len(idx), 6, MAX_LENDTH, 3))
        if not self.predict:
            target = np.zeros((len(idx), self.y.shape[1]))
        
        for i, obj in enumerate(idx):
            ndar = self.X[self.X['object_id'] == obj][['mjd', 'passband', 'flux', 'flux_err', 'detected']]
            stats = get_stats(ndar) # for defining step size
            data = to_binned_timeseries(ndar.values, np.median(stats[:, 1]))
            
            # adding zeros to MAX_LENDTH
            if data.shape[1] < MAX_LENDTH:
                data = np.concatenate((data, np.zeros((6, MAX_LENDTH - data.shape[1], 3))), axis = 1)
                
            batch[i] = data
            if not self.predict:
                target[i] = self.y.loc[obj].values

        if self.predict:
            return batch
        else:
            return batch, target
        
    def on_epoch_end(self):
        if not self.predict:
            np.random.shuffle(self.index)
        
    def __len__(self):
        if self.predict:
            return int(np.ceil(len(self.index) / self.batch_size))
        else:
            return int(len(self.index) / self.batch_size)
def get_model(class_cnt, input_shape, dropout = .5):
    inputs = keras.layers.Input(shape = input_shape)
    
    # Convolutional block
    
    x = inputs
    
    x = keras.layers.Conv2D(filters = 8, kernel_size = 1, padding = 'same', use_bias = False, 
                            kernel_initializer = keras.initializers.he_normal(seed = 0),
                            kernel_regularizer = keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization(momentum = 0.9)(x)
    x = keras.layers.LeakyReLU(alpha = .3)(x)
    
    x = keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same', use_bias = False, 
                            kernel_initializer = keras.initializers.he_normal(seed = 0),
                            kernel_regularizer = keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization(momentum = 0.9)(x)
    x = keras.layers.LeakyReLU(alpha = .2)(x)
    
    x = keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', use_bias = False, 
                            kernel_initializer = keras.initializers.he_normal(seed = 0),
                            kernel_regularizer = keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization(momentum = 0.9)(x)
    x = keras.layers.LeakyReLU(alpha = .1)(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense block
    
    x = keras.layers.Flatten()(x)
    
    x = keras.layers.Dense(class_cnt * 4, 
                           kernel_initializer = keras.initializers.he_normal(seed = 0),
                           kernel_regularizer = keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha = 0)(x)
    
    x = keras.layers.Dropout(dropout)(x)
    
    x = keras.layers.Dense(class_cnt * 2, 
                           kernel_initializer = keras.initializers.he_normal(seed = 0),
                           kernel_regularizer = keras.regularizers.l2(0.01))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha = 0)(x)
    
    x = keras.layers.Dropout(dropout)(x)
    
    outputs = keras.layers.Dense(class_cnt, 
                                 kernel_initializer = keras.initializers.he_normal(seed = 0), 
                                 activation = 'softmax')(x)
    
    return keras.Model(inputs, outputs)
OPTIMIZER = keras.optimizers.Adam(lr = 0.0005)

# The behavior of this metric completely coincides with the custom function. Only absolute values differ.
LOSS = 'categorical_crossentropy' 

METRICS = ['categorical_accuracy']
# Cross-validation for Keras model
def cv_scores(num_folds, classes, model_file_prefix, 
              X_train, y_train, 
              early_stopping = -1,
              n_epoch = 50, batch_size = 32, rs = 0):
    
    def lr_schedule_cosine(x):
        return .001 * (np.cos(np.pi * x / n_epoch) + 1.) / 2
    
    warnings.simplefilter('ignore')
    
    print("Starting cross-validation at {} with random_state {}".format(time.ctime(), rs))

    folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = rs)
        
    # Create arrays to store results
    train_pred = pd.DataFrame(columns = classes, index = y_train['object_id'])
    valid_pred = pd.DataFrame(columns = classes, index = y_train['object_id'])
    
    y = pd.get_dummies(y_train.set_index('object_id')['target']).reset_index()
        
    histories = {}

    # Cross-validation cycle
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(np.zeros(y_train.shape[0]), 
                                                                y_train.set_index('object_id'))):
        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))
        
        # Preparing data
        train_y = y.iloc[train_idx]
        train_objects = train_y['object_id'].values
        train_x = X_train[X_train['object_id'].isin(train_objects)]
        
        valid_y = y.iloc[valid_idx]
        valid_objects = valid_y['object_id'].values
        valid_x = X_train[X_train['object_id'].isin(valid_objects)]
        
        # Defining new model
        train_gen = BatchGenerator(train_x, train_y.set_index('object_id'), batch_size = batch_size)
        valid_gen = BatchGenerator(valid_x, valid_y.set_index('object_id'), batch_size = batch_size)
        
        model = get_model(len(classes), (6, MAX_LENDTH, 3))
            
        model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = METRICS)
        
        model_file = model_file_prefix + '_fold_' + str(n_fold) + '.h5'
        callbacks = [
                keras.callbacks.LearningRateScheduler(lr_schedule_cosine),
                keras.callbacks.ModelCheckpoint(filepath = model_file, 
                                                monitor = 'val_loss', 
                                                save_best_only = True, save_weights_only = True)
        ]
        
        if early_stopping > 0:
            callbacks.append(keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = early_stopping))

        # Fitting model
        model.fit_generator(train_gen, validation_data = valid_gen, callbacks = callbacks, epochs = n_epoch)
        
        histories[n_fold] = model.history.history
        
        # Prediction for train and valid data

        train_gen = BatchGenerator(train_x, None, batch_size = 1, predict = True)
        valid_gen = BatchGenerator(valid_x, None, batch_size = 1, predict = True)
        
        model.load_weights(model_file)
        
        train_pred.loc[train_objects] = pd.DataFrame(model.predict_generator(train_gen), 
                                                  columns = classes, index = train_objects) 
        valid_pred.loc[valid_objects] = pd.DataFrame(model.predict_generator(valid_gen), 
                                                  columns = classes, index = valid_objects)
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()
        
    return train_pred, valid_pred, histories
# Custom score function for galactic subset

def weighted_multiclass_logloss(y_true, y_pred):
    class_weights = [1, 1, 1, 1, 1]
    
    y_pred_clip = np.clip(a = y_pred, a_min = 1e-15, a_max = 1 - 1e-15)
    
    loss = np.sum(y_true * y_pred_clip.applymap(np.log), axis = 0)
    loss /= np.sum(y_true, axis = 0)
    loss *= class_weights
    return -(np.sum(loss) / np.sum(class_weights))
# Function for visualizing history

def plot_history(hist):
    n_folds = len(hist)
    _, axes = plt.subplots(n_folds, 2, figsize = (25, 7 * n_folds))
    
    for row in range(n_folds):
        n_epoch = len(hist[row]["loss"])
        axes[row, 0].plot(range(1, n_epoch + 1), hist[row]["loss"], label = "Train loss")
        axes[row, 0].plot(range(1, n_epoch + 1), hist[row]["val_loss"], label = "Valid loss")
        axes[row, 0].legend()
    
        axes[row, 1].plot(range(1, n_epoch + 1), hist[row]['categorical_accuracy'], label = 'Train accuracy')
        axes[row, 1].plot(range(1, n_epoch + 1), hist[row]['val_categorical_accuracy'], label = 'Valid accuracy')
        axes[row, 1].legend()
gal_train_pred, gal_valid_pred, gal_histories = cv_scores(num_folds = 2, 
                                                          n_epoch = 100, 
                                                          model_file_prefix = 'galactic',
                                                          classes = galactic_classes, 
                                                          X_train = galactic_set, 
                                                          y_train = galactic_target)
plot_history(gal_histories)
gal_train_pred.head()
gal_valid_pred.head()
y = pd.get_dummies(galactic_target.set_index('object_id')['target'])
y.head()
print('Custom score for train: ', weighted_multiclass_logloss(y, gal_train_pred))
print('Custom score for valid: ', weighted_multiclass_logloss(y, gal_valid_pred))