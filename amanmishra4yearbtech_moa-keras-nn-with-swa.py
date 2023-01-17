
# popular EDA libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn import *
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
import statsmodels.api as sm
from tqdm.notebook import tqdm
import gc

# model related libraries



import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
from keras.layers import Dense, BatchNormalization, Input
from keras.models import Model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, log_loss 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
tf.random.set_seed(2) # for reproducible results



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
train = pd.read_csv('../input/lish-moa/train_features.csv')
targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
targets_non_scored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
test = pd.read_csv('../input/lish-moa/test_features.csv')
sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

# let's see few rows of both train and test
train.head()

# for test data
test.head()
# targets corresponding to train data
targets_scored.head()
# Shapes of data

print('train data shape - ',train.shape)
print('test data shape - ',test.shape)
print('Different MoA labels - ',targets_scored.shape[1]-1)
# printing the unique values what they are
print(np.unique(train['cp_type']))


#let's see the distribution of persons in these classes
train['cp_type'].value_counts().plot(kind='bar',figsize=[10,3])
train['cp_type'].value_counts()
# check if labels for 'ctl_vehicle' are all 0.
train1 = train.merge(targets_scored, on='sig_id')
target_cols = [c for c in targets_scored.columns if c not in ['sig_id']]
cols = target_cols + ['cp_type']
train1[cols].groupby('cp_type').sum().sum(1)
# constrcut train&test data except 'cp_type'=='ctl_vehicle' data
print(train.shape, test.shape)
train1 = train1[train1['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test1 = test[test['cp_type']!='ctl_vehicle'].reset_index(drop=True)
print(train1.shape, test1.shape)
dataset = pd.concat([train1, test1],sort=False, ignore_index= True)
dataset.head()

# the values corresponding to test data labels become NaNs after concatenation
# Applying one hot encoding to the categorical_features

obj_cols =  list(dataset.select_dtypes(include = 'object').columns[1:])   # not taking sig_id
obj_cols.append('cp_time')
dataset = pd.get_dummies(dataset, columns = obj_cols)
obj_cols
# firstly drop sig_id as the necessary test_id is already presented in test1

dataset.drop('sig_id', axis=1, inplace = True)

# slicing
target_cols = targets_scored.columns[1:]  # not icluding sig_id
feature_cols = [c for c in dataset.columns if c not in target_cols ]
x_train = dataset[feature_cols][0:len(train1)]  # columns other than cols
y_train = dataset[target_cols][0:len(train1)]
x_test = dataset[feature_cols][len(train1):]

x_train.shape

def nn_model(input_shape):

    inputs = Input(shape = input_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(1024, activation= tfa.activations.gelu)(inputs)
    x = tfa.layers.GroupNormalization(groups = 32)(x)
    x = tf.keras.layers.Dense(512, activation= tfa.activations.gelu)(x)
    x = tfa.layers.GroupNormalization(groups = 16)(x)
    x = tf.keras.layers.Dense(256, activation= tfa.activations.gelu)(x)
    x = tfa.layers.GroupNormalization(groups = 8)(x)
    outputs = tf.keras.layers.Dense(206,activation='sigmoid')(x)

    # model
    return tf.keras.models.Model(inputs,outputs)
    

N_STARTS = 6


val_pred = y_train.copy()

# making an array for rows same as x_test
test_pred = np.zeros((x_test.shape[0],206))

val_pred.loc[:, y_train.columns] = 0

for seed in tqdm(range(N_STARTS)):
    for n, (tr, tv) in enumerate(KFold(n_splits=7, random_state=seed, shuffle=True).split(y_train)):
        print(f'Fold {n}')
        
        
        model = nn_model(len(x_train.columns))
        
        # using Stochastic Weight averaging
        model.compile(optimizer=tfa.optimizers.SWA(tf.optimizers.Adam(lr = 0.001), start_averaging = 9, average_period = 6),
                      loss='binary_crossentropy', metrics = None )
        
        # Callbacks
        
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')
        # for saving best weights after each
        file_path = str(n) + "weights.best.hdf5"
        
        
        # stochastic weight averaging uses average model checkpoint
        avg_checkpoint = tfa.callbacks.AverageModelCheckpoint(filepath= file_path, monitor='val_loss', save_best_only=True,verbose=2,update_weights=True,mode='min')
        
        # early stoping
        early = EarlyStopping(monitor="val_loss", mode="min", patience= 5)
        
        history = model.fit(x_train.values[tr],
                  y_train.values[tr],
                  validation_data=(x_train.values[tv], y_train.values[tv]),
                  epochs=20, batch_size=64,
                  callbacks=[reduce_lr_loss, avg_checkpoint, early]
                 )
        
        
        # loading best weights for prediction
        model.load_weights(file_path)

        test_predict = model.predict(x_test.values)
        val_predict = model.predict(x_train.values[tv])
        
        test_pred += test_predict
        val_pred.loc[tv, y_train.columns] += val_predict
        print('')
        
        
test_pred /= ((n+1) * N_STARTS)
val_pred.loc[:, y_train.columns] /= N_STARTS        
# making predictions on test data
# firstly making a zero array of test data label shape

pred_array = np.zeros((test.shape[0],sample.shape[1]-1))

# Replacing those rows
# where 'cp_type'= !'ctl_vehicle'
pred_rows = [ c for c in sample.sig_id.values if c in test1.sig_id.values]

# collecting indexes of these rows
index_pred =  sample[sample.sig_id.isin(pred_rows)][sample.columns[1:]].index


# and now for other rows replace it with pred

c = 0
for i in list(index_pred):
    pred_array[i,:] = test_pred[c,:]
    c +=1
# submitting file

sample[sample.columns[1:]] = pred_array


sample.to_csv('submission.csv', index=False)