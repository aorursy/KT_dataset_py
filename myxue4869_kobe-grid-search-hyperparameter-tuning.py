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
## import packages and modules
import matplotlib.pyplot as plt
import seaborn as sns
import math
import h5py
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.models import Sequential
from keras.layers import Dense

# show plots inline
%matplotlib inline
## dataset path
filename = "../input/kobe-neural-network/processed.csv"
## set default figure size
figure_size = (15,10)
# set max display row number
pd.set_option('max_rows',5)
## load data
df = pd.read_csv(filename, index_col = 'shot_id')
print(df.head(3))
# split data
mask = df['shot_made_flag'].isna()
predict_df = df[mask]
train_df = df[~mask]
## matrix
Y = train_df['shot_made_flag'].as_matrix()
Y = Y.reshape(Y.shape[0],1) 
X = train_df.drop(['shot_made_flag'], axis=1).as_matrix()
predict_Y = predict_df['shot_made_flag'].as_matrix()
predict_Y = predict_Y.reshape(predict_Y.shape[0],1)
predict_X = predict_df.drop(['shot_made_flag'], axis=1).as_matrix()
import keras
from keras import losses, metrics, optimizers
from keras.initializers import Initializer
from keras.layers import Activation,Dropout
print(dir(losses))
print(dir(optimizers))
print(dir(keras.initializers))
## hyperparameters
np.random.seed(1)
activations = ['relu','tanh']
kernel_inits = ['random_normal', 'zeros', 'ones']
losses = ['binary_crossentropy']
optimizers = ['Adam', 'Nadam', 'RMSprop', 'SGD']
drops = [0,0.1,0.2,0.4,0.6,0.8]

## create model
def create_model(input_dim, drop=0.2, activation = 'relu', kernel_init = 'random_uniform', 
                 loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy']):
    # define a model
    model = Sequential()
    model.add(Dense(80, input_dim=input_dim, kernel_initializer=kernel_init, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(50, kernel_initializer=kernel_init, activation=activation))
    model.add(Dropout(drop))
    model.add(Dense(1, kernel_initializer=kernel_init, activation='sigmoid'))
    # compile a model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
# fixed hyperparameters
_,dim = X.shape 
## grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
model = KerasClassifier(build_fn=create_model, 
                        epochs=10, 
                        batch_size=1024,
                        verbose=0)
param_grid = dict(input_dim=[dim], drop=drops, activation=activations, loss=losses, optimizer=optimizers, kernel_init=kernel_inits)
print(param_grid)
grid = GridSearchCV(cv=4, estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, Y)

# summarize results
print("Parameters of the best model: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print('Finished.')
## capture the best params
params = grid_result.best_params_
## create the model with the best params found
model = create_model(input_dim = dim,
                     drop=params['drop'],
                     loss=params['loss'],
                     kernel_init=params['kernel_init'],
                     activation=params['activation'],
                     optimizer=params['optimizer'])
print(params)
print(model.summary())
## Fit the model
# early stop condition
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=10**(-6))
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='auto')
# fit
history = model.fit(X, Y, validation_split=0.2, verbose=1, epochs=100, batch_size=16, callbacks=[es, reduce_lr])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'], label = 'train')
plt.plot(history.history['val_acc'], label = 'test')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper left')
plt.show()
## predict
predict_Y = model.predict(predict_X)
submission = pd.DataFrame()
submission['shot_id'] = df[mask].index
submission['shot_made_flag'] = predict_Y.reshape(predict_Y.shape[0]).tolist()
## submit
print(submission.head(3))
submission.sort_values('shot_id',inplace=True)
submission.to_csv("submission.csv",index=False)
