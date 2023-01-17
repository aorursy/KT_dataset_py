import tensorflow as tf

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import seaborn as sns

import time

import gc

import sys



print(f"Tensorflow Version: {tf.__version__}")

print(f"Pandas Version: {pd.__version__}")

print(f"Numpy Version: {np.__version__}")

print(f"System Version: {sys.version}")



mpl.rcParams['figure.figsize'] = (17, 5)

mpl.rcParams['axes.grid'] = False

sns.set_style("whitegrid")



notebookstart= time.time()
# Data Loader Parameters

BATCH_SIZE = 256

BUFFER_SIZE = 10000

TRAIN_SPLIT = 300000



# LSTM Parameters

EVALUATION_INTERVAL = 200

EPOCHS = 4

PATIENCE = 5



# Reproducibility

SEED = 13

tf.random.set_seed(SEED)
zip_path = tf.keras.utils.get_file(

    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',

    fname='jena_climate_2009_2016.csv.zip',

    extract=True)

csv_path, _ = os.path.splitext(zip_path)



df = pd.read_csv(csv_path)

print("DataFrame Shape: {} rows, {} columns".format(*df.shape))

display(df.head())
def univariate_data(dataset, start_index, end_index, history_size, target_size):

    data = []

    labels = []



    start_index = start_index + history_size

    if end_index is None:

        end_index = len(dataset) - target_size



    for i in range(start_index, end_index):

        indices = range(i-history_size, i)

        # Reshape data from (history_size,) to (history_size, 1)

        data.append(np.reshape(dataset[indices], (history_size, 1)))

        labels.append(dataset[i+target_size])

        

    return np.array(data), np.array(labels)
uni_data = df['T (degC)']

uni_data.index = df['Date Time']

uni_data.head()
uni_data.plot(subplots=True)

plt.show()

uni_data = uni_data.values
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()

uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std
univariate_past_history = 20

univariate_future_target = 0



x_train_uni, y_train_uni = univariate_data(dataset=uni_data,

                                           start_index=0,

                                           end_index=TRAIN_SPLIT,

                                           history_size=univariate_past_history,

                                           target_size=univariate_future_target)

x_val_uni, y_val_uni = univariate_data(dataset=uni_data,

                                       start_index=TRAIN_SPLIT,

                                       end_index=None,

                                       history_size=univariate_past_history,

                                       target_size=univariate_future_target)
print("In:")

print(uni_data.shape)

print(uni_data[:5])



print("\nOut")

print(x_train_uni.shape)





print(x_train_uni.shape[0] / uni_data.shape[0])
print ('Single window of past history. Shape: {}'.format(x_train_uni[0].shape))

print (x_train_uni[0])

print ('\n Target temperature to predict. Shape: {}'.format(y_train_uni[0].shape))

print (y_train_uni[0])
def create_time_steps(length):

    return list(range(-length, 0))
def show_plot(plot_data, delta, title):

    labels = ['History', 'True Future', 'Model Prediction']

    marker = ['.-', 'rx', 'go']

    time_steps = create_time_steps(plot_data[0].shape[0])

    if delta:

        future = delta

    else:

        future = 0



    plt.title(title)

    for i, x in enumerate(plot_data):

        if i:

            plt.plot(future, plot_data[i], marker[i], markersize=10,

                   label=labels[i])

        else:

            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])

        plt.legend()

        plt.xlim([time_steps[0], (future+5)*2])

        plt.xlabel('Time-Step')

    

    return plt
show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
def baseline(history):

    return np.mean(history)
show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,

           'Baseline Prediction Example')
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))

train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))

val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
x_train_uni.shape
simple_lstm_model = tf.keras.models.Sequential([

    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),

    tf.keras.layers.Dense(1)

])



simple_lstm_model.compile(optimizer='adam', loss='mae')
for x, y in val_univariate.take(1):

    print(simple_lstm_model.predict(x).shape)
early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)

simple_lstm_model.fit(train_univariate,

                      epochs=EPOCHS,

                      steps_per_epoch=EVALUATION_INTERVAL,

                      validation_data=val_univariate,

                      callbacks=[early_stopping],

                      validation_steps=50)
for x, y in val_univariate.take(3):

    plot = show_plot([x[0].numpy(), y[0].numpy(),

                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')

    plot.show()
del simple_lstm_model, val_univariate, train_univariate
features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]

features.index = df['Date Time']

features.head()
features.plot(subplots=True)
dataset = features.values

data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)

data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std
display(pd.DataFrame(dataset, columns = features.columns, index= features.index).head())
def multivariate_data(dataset, target, start_index, end_index, history_size,

                      target_size, step, single_step=False):

    data = []

    labels = []



    start_index = start_index + history_size

    if end_index is None:

        end_index = len(dataset) - target_size



    for i in range(start_index, end_index):

        indices = range(i-history_size, i, step)

        data.append(dataset[indices])



        if single_step:

            labels.append(target[i+target_size])

        else:

            labels.append(target[i:i+target_size])



    return np.array(data), np.array(labels)
past_history = 720

future_target = 72

STEP = 6



x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,

                                                   TRAIN_SPLIT, past_history,

                                                   future_target, STEP,

                                                   single_step=True)

x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],

                                               TRAIN_SPLIT, None, past_history,

                                               future_target, STEP,

                                               single_step=True)
print(x_train_single.shape)

print ('Single window of past history : {}'.format(x_train_single[0].shape))

print(x_train_single.shape[-2:])
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))

train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))

val_data_single = val_data_single.batch(BATCH_SIZE).repeat()
single_step_model = tf.keras.models.Sequential()

single_step_model.add(tf.keras.layers.LSTM(32,

                                           input_shape=x_train_single.shape[-2:]))

single_step_model.add(tf.keras.layers.Dense(1))



single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
for x, y in val_data_single.take(1):

    print(single_step_model.predict(x).shape)
print(f"Evaluation Threshold: {EVALUATION_INTERVAL}",

      f"Epochs: {EPOCHS}", sep="\n")



early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)

single_step_history = single_step_model.fit(train_data_single,

                                            epochs=EPOCHS,

                                            steps_per_epoch=EVALUATION_INTERVAL,

                                            validation_data=val_data_single,

                                            callbacks=[early_stopping],

                                            validation_steps=50)
def plot_train_history(history, title):

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(len(loss))



    plt.figure()



    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'r', label='Validation loss')

    plt.title(title)

    plt.legend()



    plt.show()
plot_train_history(single_step_history,

                   'Single Step Training and validation loss')
for x, y in val_data_single.take(3):

    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),

                    single_step_model.predict(x)[0]], 12,

                   'Single Step Prediction')

    plot.show()
del single_step_history, val_data_single, train_data_single
past_history = 720

future_target = 72

STEP = 6



x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,

                                                 TRAIN_SPLIT, past_history,

                                                 future_target, STEP)

x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],

                                             TRAIN_SPLIT, None, past_history,

                                             future_target, STEP)
print (x_train_multi.shape,

       y_train_multi.shape,

       'Single window of past history : {}'.format(x_train_multi[0].shape),

       'Target temperature to predict : {}'.format(y_train_multi[0].shape),

       sep='\n')
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))

train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
def multi_step_plot(history, true_future, prediction):

    plt.figure(figsize=(18, 6))

    num_in = create_time_steps(len(history))

    num_out = len(true_future)



    plt.plot(num_in, np.array(history[:, 1]), label='History')

    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',

           label='True Future')

    if prediction.any():

        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',

                 label='Predicted Future')

    plt.legend(loc='upper left')

    plt.show()
for x, y in train_data_multi.take(1):

    multi_step_plot(x[0], y[0], np.array([0]))
multi_step_model = tf.keras.models.Sequential()

multi_step_model.add(tf.keras.layers.LSTM(32,

                                          return_sequences=True,

                                          input_shape=x_train_multi.shape[-2:]))

multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))

multi_step_model.add(tf.keras.layers.Dense(72))



multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

print(multi_step_model.summary())
for x, y in val_data_multi.take(1):

    print (multi_step_model.predict(x).shape)
early_stopping = EarlyStopping(monitor='val_loss', patience = 3, restore_best_weights=True)

multi_step_history = multi_step_model.fit(train_data_multi,

                                          epochs=EPOCHS,

                                          steps_per_epoch=EVALUATION_INTERVAL,

                                          validation_data=val_data_multi,

                                          validation_steps=EVALUATION_INTERVAL,

                                          callbacks=[early_stopping])
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')
for x, y in val_data_multi.take(3):

    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
del multi_step_model, val_data_multi, train_data_multi

_ = gc.collect()
from tensorflow.keras.layers import *

from tensorflow.keras.models import Sequential
def multivariate_multioutput_data(dataset, target, start_index, end_index, history_size,

                      target_size, step, single_step=False):

    data = []

    labels = []



    start_index = start_index + history_size

    if end_index is None:

        end_index = len(dataset) - target_size



    for i in range(start_index, end_index):

        indices = range(i-history_size, i, step)

        data.append(dataset[indices])



        if single_step:

            labels.append(target[i+target_size])

        else:

            labels.append(target[i:i+target_size])



    return np.array(data)[:,:,:,np.newaxis,np.newaxis], np.array(labels)[:,:,:,np.newaxis,np.newaxis]



def multi_step_output_plot(history, true_future, prediction):

    plt.figure(figsize=(18, 6))

    num_in = create_time_steps(len(history))

    num_out = len(true_future)

    

    for i, (var, c) in enumerate(zip(features.columns[:2], ['b','r'])):

        plt.plot(num_in, np.array(history[:, i]), c, label=var)

        plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,i]), c+'o', markersize=5, alpha=0.5,

               label=f"True {var.title()}")

        if prediction.any():

            plt.plot(np.arange(num_out)/STEP, np.array(prediction[:,i]), '*', markersize=5, alpha=0.5,

                     label=f"Predicted {var.title()}")

    

    plt.legend(loc='upper left')

    plt.show()
future_target = 72

x_train_multi, y_train_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:,:2], 0,

                                                 TRAIN_SPLIT, past_history,

                                                 future_target, STEP)

x_val_multi, y_val_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:, :2],

                                             TRAIN_SPLIT, None, past_history,

                                             future_target, STEP)
print (x_train_multi.shape,

       y_train_multi.shape,

       x_val_multi.shape,

       y_val_multi.shape,

       'Single window of past history : {}'.format(x_train_multi[0].shape),

       'Target temperature to predict : {}'.format(y_train_multi[0].shape),

       sep='\n')
BATCH_SIZE = 128



train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))

train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
for x, y in val_data_multi.take(10):

    multi_step_output_plot(np.squeeze(x[0]), np.squeeze(y[0]), np.array([0]))
def build_model(input_timesteps, output_timesteps, num_links, num_inputs):

    # COPY PASTA

    # https://github.com/niklascp/bus-arrival-convlstm/blob/master/jupyter/ConvLSTM_3x15min_10x64-5x64-10x64-5x64-Comparison.ipynb

    

    model = Sequential()

    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_inputs, 1, 1)))

    model.add(ConvLSTM2D(name ='conv_lstm_1',

                         filters = 64, kernel_size = (10, 1),                       

                         padding = 'same', 

                         return_sequences = True))

    

    model.add(Dropout(0.30, name = 'dropout_1'))

    model.add(BatchNormalization(name = 'batch_norm_1'))



    model.add(ConvLSTM2D(name ='conv_lstm_2',

                         filters = 64, kernel_size = (5, 1), 

                         padding='same',

                         return_sequences = False))

    

    model.add(Dropout(0.20, name = 'dropout_2'))

    model.add(BatchNormalization(name = 'batch_norm_2'))

    

    model.add(Flatten())

    model.add(RepeatVector(output_timesteps))

    model.add(Reshape((output_timesteps, num_inputs, 1, 64)))

    

    model.add(ConvLSTM2D(name ='conv_lstm_3',

                         filters = 64, kernel_size = (10, 1), 

                         padding='same',

                         return_sequences = True))

    

    model.add(Dropout(0.20, name = 'dropout_3'))

    model.add(BatchNormalization(name = 'batch_norm_3'))

    

    model.add(ConvLSTM2D(name ='conv_lstm_4',

                         filters = 64, kernel_size = (5, 1), 

                         padding='same',

                         return_sequences = True))

    

    model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))

    model.add(Dense(units=1, name = 'dense_2', activation = 'linear'))



#     optimizer = RMSprop() #lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)

#     optimizer = tf.keras.optimizers.Adam(0.1)

    optimizer = tf.keras.optimizers.RMSprop(lr=0.004, clipvalue=1.0)

    model.compile(loss = "mse", optimizer = optimizer, metrics = ['mae', 'mse'])

    return model
future_target = 72

x_train_multi, y_train_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:,:2], 0,

                                                 TRAIN_SPLIT, past_history,

                                                 future_target, STEP)

x_val_multi, y_val_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:, :2],

                                             TRAIN_SPLIT, None, past_history,

                                             future_target, STEP)



BATCH_SIZE = 128



train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))

train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
EPOCHS = 40

steps_per_epoch = 350

validation_steps = 500



modelstart = time.time()

early_stopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)

model = build_model(x_train_multi.shape[1], future_target, y_train_multi.shape[2], x_train_multi.shape[2])

print(model.summary())



# Train

print("\nTRAIN MODEL...")

history = model.fit(train_data_multi,

                    epochs = EPOCHS,

                    validation_data=val_data_multi,

                    steps_per_epoch=steps_per_epoch,

                    validation_steps=validation_steps,

                    verbose=1,

                    callbacks=[early_stopping])

model.save('multi-output-timesteps.h5')

print("\nModel Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
plot_train_history(history, 'Multi-Step, Multi-Output Training and validation loss')
for x, y in val_data_multi.take(10):

    multi_step_output_plot(np.squeeze(x[0]), np.squeeze(y[0]), np.squeeze(model.predict(x[0][np.newaxis,:,:,:,:])))
def build_model(input_timesteps, output_timesteps, num_links, num_inputs):    

    model = Sequential()

    model.add(BatchNormalization(name = 'batch_norm_0', input_shape = (input_timesteps, num_inputs, 1, 1)))

    model.add(ConvLSTM2D(name ='conv_lstm_1',

                         filters = 64, kernel_size = (10, 1),                       

                         padding = 'same', 

                         return_sequences = False))

    

    model.add(Dropout(0.30, name = 'dropout_1'))

    model.add(BatchNormalization(name = 'batch_norm_1'))



#     model.add(ConvLSTM2D(name ='conv_lstm_2',

#                          filters = 64, kernel_size = (5, 1), 

#                          padding='same',

#                          return_sequences = False))

    

#     model.add(Dropout(0.20, name = 'dropout_2'))

#     model.add(BatchNormalization(name = 'batch_norm_2'))

    

    model.add(Flatten())

    model.add(RepeatVector(output_timesteps))

    model.add(Reshape((output_timesteps, num_inputs, 1, 64)))

    

#     model.add(ConvLSTM2D(name ='conv_lstm_3',

#                          filters = 64, kernel_size = (10, 1), 

#                          padding='same',

#                          return_sequences = True))

    

#     model.add(Dropout(0.20, name = 'dropout_3'))

#     model.add(BatchNormalization(name = 'batch_norm_3'))

    

    model.add(ConvLSTM2D(name ='conv_lstm_4',

                         filters = 64, kernel_size = (5, 1), 

                         padding='same',

                         return_sequences = True))

    

    model.add(TimeDistributed(Dense(units=1, name = 'dense_1', activation = 'relu')))

    model.add(Dense(units=1, name = 'dense_2'))



#     optimizer = RMSprop() #lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)

#     optimizer = tf.keras.optimizers.Adam(0.1)

    optimizer = tf.keras.optimizers.RMSprop(lr=0.003, clipvalue=1.0)

    model.compile(loss = "mse", optimizer = optimizer, metrics = ['mae', 'mse'])

    return model
# Extend Prediction Window..

future_target = 144

x_train_multi, y_train_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:,:2], 0,

                                                 TRAIN_SPLIT, past_history,

                                                 future_target, STEP)

x_val_multi, y_val_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:, :2],

                                             TRAIN_SPLIT, None, past_history,

                                             future_target, STEP)



BATCH_SIZE = 128



train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))

train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()



val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
EPOCHS = 30

steps_per_epoch = 350

validation_steps = 500



modelstart = time.time()

early_stopping = EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)

model = build_model(x_train_multi.shape[1], future_target, y_train_multi.shape[2], x_train_multi.shape[2])

print(model.summary())



# Train

print("\nTRAIN MODEL...")

history = model.fit(train_data_multi,

                    epochs = EPOCHS,

                    validation_data=val_data_multi,

                    steps_per_epoch=steps_per_epoch,

                    validation_steps=validation_steps,

                    verbose=1,

                    callbacks=[early_stopping])

model.save('multi-output-timesteps.h5')

print("\nModel Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
plot_train_history(history, 'Multi-Step, Multi-Output Training and validation loss')
for x, y in val_data_multi.take(10):

    multi_step_output_plot(np.squeeze(x[0]), np.squeeze(y[0]), np.squeeze(model.predict(x[0][np.newaxis,:,:,:,:])))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))