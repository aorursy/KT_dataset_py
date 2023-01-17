# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pathlib

from sklearn.preprocessing import StandardScaler, RobustScaler

import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



import warnings

warnings.filterwarnings("ignore")

warnings.filterwarnings(action="ignore",category=DeprecationWarning)

warnings.filterwarnings(action="ignore",category=FutureWarning)

print(f'tf={tf.__version__}, keras={keras.__version__}')
SEED = 31

EPOCHS = 1000

BATCH_SIZE = 1000

VALIDATION_SPLIT = 0.01

TARGET = 'scalar_coupling_constant'

PREDICTORS = [

    'molecule_atom_index_0_dist_mean_div',

    'molecule_atom_index_0_dist_max_div',

    'molecule_atom_index_1_dist_max_div',

    'molecule_atom_index_0_dist_std_div',

    'molecule_atom_index_0_dist_min_div',

    'molecule_atom_index_1_dist_mean_div',

    'molecule_atom_index_1_dist_std_div',

    'molecule_atom_1_dist_std_diff',

    'molecule_atom_index_0_dist_std_diff',

    'molecule_atom_index_0_dist_mean_diff',

    'molecule_atom_index_1_dist_max_diff',

    'molecule_atom_index_0_dist_max_diff',

    'molecule_type_0_dist_std_diff',

    'molecule_atom_index_1_dist_mean_diff',

    'molecule_atom_index_1_dist_std_diff',

    'molecule_atom_1_dist_min_div',

    'molecule_atom_1_dist_min_diff',

    'type_0',

    'type_1',

    'molecule_type_dist_min',

    'molecule_type_dist_mean',

    'molecule_type_0_dist_std',

    'dist_to_type_1_mean',

    'dist',

    'molecule_type_dist_max',

    'dist_x',

    'dist_y',

    'dist_z'

]
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

seed_everything(SEED)
file_folder = '../input/champs-scalar-coupling-preprocess'

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

print(f'train={train.shape}, test={test.shape}')
#train = train_whole.sample(frac=0.99)

#validation = train_whole.drop(train.index)

#print(f'train={train.shape}, validation={validation.shape}')
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    maes = np.log(maes.map(lambda x: max(x, floor)))

    print(maes)

    return maes.mean()
y_train = train[TARGET]

x_train = train[PREDICTORS]

x_test = test[PREDICTORS]

scaler = RobustScaler().fit(x_train.values)

norm = scaler.transform(x_train.values)

x_train = pd.DataFrame(norm, index=x_train.index, columns=x_train.columns)

norm = scaler.transform(x_test.values)

x_test = pd.DataFrame(norm, index=x_test.index, columns=x_test.columns)
def build_model():

  model = keras.Sequential([

    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(x_train.keys())]),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(64, activation=tf.nn.relu),

    layers.Dense(1)

  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_absolute_error',

                optimizer=optimizer,

                metrics=['mean_absolute_error', 'mean_squared_error'])

  return model





model = build_model()

model.summary()
# sanity check model is producing output of desired type and shape



example_batch = x_train[:10]

example_result = model.predict(example_batch)

example_result
%%time



# Display training progress by printing a single dot for each completed epoch

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')





# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)    



history = model.fit(

  x_train, y_train,

  epochs=EPOCHS, validation_split = VALIDATION_SPLIT, verbose=0, batch_size=BATCH_SIZE,

  callbacks=[early_stop, PrintDot()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error [MPG]')

  plt.plot(hist['epoch'], hist['mean_absolute_error'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],

           label = 'Val Error')

  plt.ylim([0,5])

  plt.legend()



  #plt.figure()

  #plt.xlabel('Epoch')

  #plt.ylabel('Mean Square Error [$MPG^2$]')

  #plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')

  #plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')

  #plt.ylim([0,20])

  #plt.legend()

  plt.show()





plot_history(history)
#model = build_model()



# The patience parameter is the amount of epochs to check for improvement

#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



#history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,

 #                   validation_split = VALIDATION_SPLIT, verbose=0, callbacks=[early_stop, PrintDot()])



#plot_history(history)
#loss, mae, mse = model.evaluate(test, y_test, verbose=0)



#print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
y_pred_train = model.predict(x_train, batch_size=BATCH_SIZE).flatten()

gmlm = group_mean_log_mae(y_train, y_pred_train, train['type'])

print('group_mean_log_mae={}'.format(gmlm))
preds = model.predict(x_test, batch_size=BATCH_SIZE).flatten()

print(preds)
submission = pd.DataFrame({'id': test['id'], 'scalar_coupling_constant': preds})

submission.head()
submission.to_csv('submission.csv', index=False)

train = pd.DataFrame({'id': train['id'], 'type': train['type'], TARGET: train[TARGET], 'pred': y_pred_train})

train.to_csv('train.csv', index=False)

print(os.listdir("."))