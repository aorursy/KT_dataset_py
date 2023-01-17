import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# ML

import tensorflow as tf

print('TensorFlow version: ', tf.__version__)

from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Other

import zipfile



# Clear Session

tf.keras.backend.clear_session()

print('TensorFlow session cleared.')
# Functions to be used in this section.



# plot_ID plots most important features of a profile ID (= load cycle)

def plot_ID(df, PID, path = None):

    

    vis_df = df.where(data_df['profile_id'] == PID).dropna()



    # Change index to time vector (sample rate is 2 Hz)

    vis_df.index = np.arange(0, 2*len(vis_df['profile_id']), 2)

    vis_df.index.name = 'time'

    

    fig, axes = plt.subplots(5, 1, sharex = True, sharey = True, figsize=(15, 10))



    # Motor speed

    axes[0].plot(vis_df.index, vis_df['motor_speed'])

    axes[0].set_title('speed')

    

    # Torque

    axes[1].plot(vis_df.index, vis_df['torque'])

    axes[1].set_title('torque')



    # Current (d/q)

    axes[2].plot(vis_df.index, vis_df['i_d'])

    axes[2].plot(vis_df.index, vis_df['i_q'])

    axes[2].legend(['i_d', 'i_q'], loc='right')

    axes[2].set_title('current d/q')

    

    # Voltage (d/q)

    axes[3].plot(vis_df.index, vis_df['u_d'])

    axes[3].plot(vis_df.index, vis_df['u_q'])

    axes[3].set_title('voltage d/q');

    axes[3].legend(['u_d', 'u_q'], loc='right')

    

    # Relevant Temperatures (Ambient, Coolant, PM, Winding)

    axes[4].plot(vis_df.index, vis_df['ambient'])

    axes[4].plot(vis_df.index, vis_df['coolant'])

    axes[4].plot(vis_df.index, vis_df['pm'])

    axes[4].plot(vis_df.index, vis_df['stator_winding'])

    axes[4].legend(['amb', 'cool', 'pm', 'wdg'], loc='right')

    axes[4].set_title('temperatures')

    axes[4].set_xlabel('time')



    plt.subplots_adjust(hspace=0.5);

    fig.suptitle('Profile_ID ' + str(PID), fontsize="x-large", fontweight='bold')

    

    # if path is given, figure is saved to directory instead of printed

    if path:

        fig.savefig(path + '/Profile_' + str(PID) + '.png'), 

                    #dpi=f.dpi, bbox_inches='tight', bbox_extra_artists=[ttl])

        plt.close(fig)



        

# plot_ID_len plots the length of the unique load cycles.

def plot_ID_len(df):

    

    vis_df = df.groupby(['profile_id'])

    vis_df = vis_df.size().sort_values().rename('length').reset_index()

    ordered_ids = vis_df.profile_id.values.tolist()

    fig = plt.figure(figsize=(17, 5))

    sns.barplot(y='length', x='profile_id', data=vis_df, order=ordered_ids)

    tcks = plt.yticks(2*3600*np.arange(1, 8), [f'{a} hrs' for a in range(1, 8)]) # 2Hz sample rate

    print('Max load profile length: {:.2f} h'.format(max(vis_df['length'])/(2*3600)))

    print('Min load profile length: {:.2f} h'.format(min(vis_df['length'])/(2*3600)))
# Import the dataset and print basic info



data_df = pd.read_csv('/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv')



print('---- Dataset Info ----')

data_df.info()

print('----------------')

print('NaNs existing in dataset is: ', data_df.isnull().values.any())

print('----------------')

data_df.head(10)
# Visualize load cycles (= profile_id)



# Profile_ID is a unique ID for each measurement.

IDs = np.array(data_df['profile_id'].unique())

print('Profile ID count: ', len(IDs))

print('Unique Profile IDs (= load cycles):\n', IDs)



# Let's visualize a random load cycle

PID = np.random.choice(IDs)

print('Plotting Profile ID: ', PID, '....')

plot_ID(data_df, PID)
# Manually examine all load cycles by saving figures to disk.

    

SaveAllCycles = False

    

if SaveAllCycles:    

    path = os.getcwd()



    if not os.path.exists(path + '/cycles'):

        os.mkdir('/kaggle/working/cycles')



    #for PID in data_df['profile_id'].unique():

    #    plot_ID(data_df, PID, path = '/kaggle/working/cycles')



    myzip = zipfile.ZipFile('/kaggle/working/cycles.zip', 'w')

    for _, _, files in os.walk('/kaggle/working/cycles'):

        [myzip.write('cycles/' + fname) for fname in files]



    print('Saved all figures and created archive.')
# Visualize load cycle lengths. Depending on the NN architecture, the profiles need to be zero-padded.



plot_ID_len(data_df)
# Hyperparameters (downsampling, train/dev/test-split, feature selection, ...)

downsample_rate = 4

n_dev = 2 

n_test = 1

window_len = 64

features = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'i_d', 'i_q', 'u_s', 'i_s', 's_el']

feature_len = len(features)

target = ['stator_winding']
# Creates single sample array of shape (1, T, n)

def get_single_sample(df, n_feat, sample_len, downsample_rate=1):

    

    # Get new length for downsampling

    new_len = int(np.floor((max_len + downsample_rate - 1) / downsample_rate) * downsample_rate)

    

    # Convert df to numpy array of shape (1, T, n)

    arr = df.to_numpy()

    arr = np.expand_dims(arr, axis=0)

    

    # Zero-pad to sample_len at the end of the array

    _arr = np.zeros((1, new_len - np.size(arr, 1), n_feat))

    sample = np.concatenate((arr, _arr), axis=1)

    

    # Get sample_weights (zero-padded elements should have zero weight)

    weights = np.concatenate((np.ones(arr.shape), np.zeros(_arr.shape)), axis=1)

    weights = weights[:,:,0]

    

    # Perform Downsampling

    dwn_sample = []

    dwn_weights = []

    for d in np.arange(1,downsample_rate+1):

        dwn_sample.append(sample[:,(-1+d)::downsample_rate,:])

        dwn_weights.append(weights[:,(-1+d)::downsample_rate])

    

    sample = np.concatenate(dwn_sample, axis=0)

    weights = np.concatenate(dwn_weights, axis=0)

        

    return sample, weights





# Creates windowed mini-batches of shape (m, T_windowed, n) with consistent order of batches. 

# This is necessary for TCNs where window_len should match the receptive field of the TCN.

# It could also be used with stateful LSTMs to implement truncated backprop through time (TBPTT).

def get_windowed_batches(X, weights, Y, window_len):

    

    if window_len >= X.shape[1]:

        raise ValueError('Window length must be less than total batch length.')

    

    # get number of splits and clip data to integer splits 

    # (the "loss" of data is affordable, mostly zero-padded data is cut away)

    T = X.shape[1]

    remainder = np.remainder(X.shape[1],window_len)

    X = X[:,:-remainder,:]

    weights = weights[:,:-remainder]

    Y = Y[:,:-remainder,:]

    n_splits = int(X.shape[1]/window_len)

    

    # split input data accordingly

    X_win = np.split(X, n_splits, axis=1)

    weights_win = np.split(weights, n_splits, axis=1)

    Y_win = np.split(Y, n_splits, axis=1)



    # reshape dimensions

    X_win = np.vstack(X_win)

    weights_win = np.vstack(weights_win)

    Y_win = np.vstack(Y_win)

    

    return X_win, weights_win, Y_win
# Derive additional features: Current and voltage magnitude & electrical apparent power



data_df['u_s'] = np.sqrt(data_df['u_d']**2 + data_df['u_q']**2)

data_df['i_s'] = np.sqrt(data_df['i_d']**2 + data_df['i_q']**2)

data_df['s_el'] = 1.5 * data_df['u_s'] * data_df['i_s']



data_df.head(10)
# Prepare Data for use with LSTMs: Data needs to be in shape (m, T, n)



# get maximum length, select features and target

max_len = data_df.groupby(['profile_id']).size().max()



# Prepare index for faster iteration

iter_df = data_df.copy() # copy increases memory use, but avoids errors when executed twice. Better solution?

iter_df['idx'] = data_df.index

iter_df.set_index(['profile_id', 'idx'], inplace = True)



# create (m, T, n) array for X_values, sample_weights and Y_values

batch_samples_X = []

batch_weights_X = []

batch_samples_Y = []



for pid in IDs:

    # select profile

    df = iter_df.loc[pid]

    # get X samples and weights

    sample, weights = get_single_sample(df[features], 10, max_len, downsample_rate)

    batch_samples_X.append(sample)

    batch_weights_X.append(weights)    

    # get Y samples

    sample, _ = get_single_sample(df[target], 1, max_len, downsample_rate)

    batch_samples_Y.append(sample)

    

X_vals = np.concatenate(batch_samples_X, axis=0)

X_weights = np.concatenate(batch_weights_X, axis=0)

Y_vals = np.concatenate(batch_samples_Y, axis=0)



print('Shape of batches')

print('X_vals:    ', X_vals.shape)

print('X_weights: ', X_weights.shape)

print('Y_vals:    ', Y_vals.shape)
# Create train-dev-test-split for LSTMs

# (when cycles are downsampled, all downsampled parts should belong to the same set)

X_train = X_vals[:-(n_dev+n_test)*downsample_rate,:,:]

X_train_weights = X_weights[:-(n_dev+n_test)*downsample_rate,:]

Y_train = Y_vals[:-(n_dev+n_test)*downsample_rate,:,:]



X_dev = X_vals[-((n_dev+n_test)*downsample_rate):-(n_test)*downsample_rate,:,:]

X_dev_weights = X_weights[-((n_dev+n_test)*downsample_rate):-(n_test)*downsample_rate,:]

Y_dev = Y_vals[-((n_dev+n_test)*downsample_rate):-(n_test)*downsample_rate,:,:]



X_test = X_vals[-((n_test)*downsample_rate):,:,:]

X_test_weights = X_weights[-((n_test)*downsample_rate):,:]

Y_test = Y_vals[-((n_test)*downsample_rate):,:,:]



print('Shape of train-test-split')

print('train (X, weights, Y): ', X_train.shape, X_train_weights.shape, Y_train.shape)

print('dev (X, weights, Y):   ', X_dev.shape, X_dev_weights.shape, Y_dev.shape)

print('test (X, weights, Y):  ', X_test.shape, X_test_weights.shape, Y_test.shape)





# Normalization / Scaling 

# >> optional, values are already at a similar scale



# EWMA filtering 

# >> optional, this de-noises the data which helps with learning and prediction

# >> however, real sensor data is also noisy, so the task is harder and more realistic with noise.
# Data preparation for TCNs is a little different comparted to LSTMs.

# TCNs will use windowed mini-batches that should match the receptive field of the TCN.



X_train_tcn, X_train_weights_tcn, Y_train_tcn = get_windowed_batches(

    X_train, X_train_weights, Y_train, window_len)

X_dev_tcn, X_dev_weights_tcn, Y_dev_tcn = get_windowed_batches(

    X_dev, X_dev_weights, Y_dev, window_len)

X_test_tcn, X_test_weights_tcn, Y_test_tcn = get_windowed_batches(

    X_test, X_test_weights, Y_test, window_len)  



print('Shape of windowed mini-batch train-test-split')

print('train (X, weights, Y): ', X_train_tcn.shape, X_train_weights_tcn.shape, Y_train_tcn.shape)

print('dev (X, weights, Y):   ', X_dev_tcn.shape, X_dev_weights_tcn.shape, Y_dev_tcn.shape)

print('test (X, weights, Y):  ', X_test_tcn.shape, X_test_weights_tcn.shape, Y_test_tcn.shape)



# Get batch_size and verify mini-batches by plotting re-stacked mini-batch

batch_size = X_train.shape[0]

sample_cycle = 49

plt.plot(Y_train[sample_cycle,:,0])

plt.plot(np.concatenate(Y_train_tcn[sample_cycle::batch_size,:,0], axis=0))

plt.legend(['full batch', 'mini batch'])

plt.title('Verification of mini-batch creation');
n_epochs = 100

lr = 0.01

lr_decay = 1e-2

dropout_rate = 0.1

spatial_dropout = 0.7

n_units = 64

n_dense_in = 32

n_dense_mid = 8

n_dense_out = 1

len_kernel = 4
# Plots the loss over all epochs and a zoom on the last 20 epochs.

def plot_learning_curves(history, descr=' '):



    # get results

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(loss)) 

    

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))



    # loss

    axes[0].plot(epochs, loss, 'r')

    axes[0].plot(epochs, val_loss, 'b')

    axes[0].set_title('Loss - Train vs. Validation')

    axes[0].legend(['Train', 'Validation'])

    axes[0].set_xlabel('Epoch')

    axes[0].set_ylabel('loss')

    

    # mse

    axes[1].plot(epochs[n_epochs-20:], loss[n_epochs-20:], 'r')

    axes[1].plot(epochs[n_epochs-20:], val_loss[n_epochs-20:], 'b')

    axes[1].set_title('Loss - Zoom to last 20 epochs')

    axes[1].legend(['Train', 'Validation'])

    axes[1].set_xlabel('Epoch')

    axes[1].set_ylabel('loss')

    axes[1].set_xticks(np.arange(n_epochs-20, n_epochs, step=2))

    

    fig.suptitle(descr, fontsize="x-large", fontweight='bold')

    

    return
simple_lstm = tf.keras.models.Sequential([

  tf.keras.layers.LSTM(n_units, return_sequences=True, input_shape=[None, feature_len]),

  tf.keras.layers.LSTM(n_units, return_sequences=True),

  tf.keras.layers.Dense(n_dense_in, activation="relu"),

  tf.keras.layers.Dropout(dropout_rate),

  tf.keras.layers.Dense(n_dense_mid, activation="relu"),

  tf.keras.layers.Dropout(dropout_rate),

  tf.keras.layers.Dense(n_dense_out),

])



simple_lstm.summary()



optimizer = tf.keras.optimizers.Adam(learning_rate=lr, 

                                     decay=lr_decay,

                                     beta_1=0.9, beta_2=0.999, amsgrad=False)



simple_lstm.compile(loss='mean_squared_error',

              optimizer=optimizer,

              metrics=['mse'],

              sample_weight_mode='temporal')



print('---- training in progress ----')



history = simple_lstm.fit(x=X_train, y=Y_train, 

                          validation_data=(X_dev, Y_dev, X_dev_weights), 

                          sample_weight=X_train_weights, 

                          epochs=n_epochs,

                          verbose=0)



print('--- done ---')
plot_learning_curves(history, descr='simple_LSTM')
inputs = tf.keras.layers.Input(shape=[None, feature_len])



# First residual LSTM block

out_1 = tf.keras.layers.LSTM(n_units, return_sequences=True)(inputs)

out_2 = tf.keras.layers.LSTM(n_units, return_sequences=True)(out_1)

add_1 = tf.keras.layers.Add()([out_1, out_2])



# Second residual LSTM block

out_3 = tf.keras.layers.LSTM(n_units, return_sequences=True)(add_1)

out_4 = tf.keras.layers.LSTM(n_units, return_sequences=True)(out_3)

add_2 = tf.keras.layers.Add()([out_3, out_4])



# Dense Layer

x = tf.keras.layers.Dense(n_dense_in, activation="relu")(add_2)

x = tf.keras.layers.Dropout(dropout_rate)(x)

x = tf.keras.layers.Dense(n_dense_mid, activation="relu")(x)

x = tf.keras.layers.Dropout(dropout_rate)(x)

y = tf.keras.layers.Dense(n_dense_out)(x)



res_lstm = tf.keras.models.Model(inputs=[inputs], outputs=y)



res_lstm.summary()



optimizer = tf.keras.optimizers.Adam(learning_rate=lr, 

                                     decay=lr_decay,

                                     beta_1=0.9, beta_2=0.999, amsgrad=False)



res_lstm.compile(loss='mean_squared_error',

              optimizer=optimizer,

              metrics=['mse'],

              sample_weight_mode='temporal')



print('---- training in progress ----')



history = res_lstm.fit(x=X_train, y=Y_train, 

                          validation_data=(X_dev, Y_dev, X_dev_weights), 

                          sample_weight=X_train_weights, 

                          epochs=n_epochs,

                          verbose=0)



print('--- done ---')
plot_learning_curves(history, descr='res_LSTM')
inputs = tf.keras.layers.Input(shape=[None, feature_len])



# First residual TCN block

x_1 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=1, activation='relu', padding='causal')(inputs)

x_1 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=2, activation='relu', padding='causal')(x_1)

x_1 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=4, activation='relu', padding='causal')(x_1)

x_1 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=8, activation='relu', padding='causal')(x_1)

x_1 = tf.keras.layers.SpatialDropout1D(rate=spatial_dropout)(x_1)

x_1_res = tf.keras.layers.Conv1D(filters=n_units, kernel_size=1, dilation_rate=2, padding='causal')(inputs)

x_1 = tf.keras.layers.Add()([x_1_res, x_1])



# Second residual TCN block

x_2 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=1, activation='relu', padding='causal')(x_1)

x_2 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=2, activation='relu', padding='causal')(x_2)

x_2 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=4, activation='relu', padding='causal')(x_2)

x_2 = tf.keras.layers.Conv1D(filters=n_units, kernel_size=len_kernel, 

           dilation_rate=8, activation='relu', padding='causal')(x_2)

x_2 = tf.keras.layers.SpatialDropout1D(rate=spatial_dropout)(x_2)

x_2_res = tf.keras.layers.Conv1D(filters=n_units, kernel_size=1, dilation_rate=2, padding='causal')(x_1)

x_2 = tf.keras.layers.Add()([x_2_res, x_2])



# Dense Layer

x_3 = tf.keras.layers.Dense(n_dense_in, activation="relu")(x_2)

x_3 = tf.keras.layers.Dropout(dropout_rate)(x_3)

x_3 = tf.keras.layers.Dense(n_dense_mid, activation="relu")(x_3)

x_3 = tf.keras.layers.Dropout(dropout_rate)(x_3)

y = tf.keras.layers.Dense(n_dense_out)(x_3)



TCN = tf.keras.models.Model(inputs=[inputs], outputs=y)



TCN.summary()



optimizer = tf.keras.optimizers.Adam(learning_rate=lr, 

                                     decay=lr_decay,

                                     beta_1=0.9, beta_2=0.999, amsgrad=False)



TCN.compile(loss='mean_squared_error',

              optimizer=optimizer,

              metrics=['mse'],

              sample_weight_mode='temporal')



print('---- training in progress ----')



history = TCN.fit(x=X_train_tcn, y=Y_train_tcn, 

                          validation_data=(X_dev_tcn, Y_dev_tcn, X_dev_weights_tcn), 

                          sample_weight=X_train_weights_tcn, 

                          epochs=n_epochs,

                          batch_size=batch_size,

                          verbose=0)



print('--- done ---')
plot_learning_curves(history, descr='TCN')
# This function makes a prediction on a given X of shape (1,Tx,1).

# The first skip_values are truncated, because initial temperature states may have large error.

# Also, only "not-zero-padded" part of the sequence is taken into account (-> sample_weight = 1).

def eval_model(model, X, Y, weights, skip_values=10, scaler=None):

    # prepare data for prediction

    end_sequence = np.where(weights==0)[1][0] # get "real" (= not-zero-padded) end of sequence

    X_pred = X[:,:end_sequence,:]

    Y_truth = Y[0,:end_sequence,0]



    # predict (and rescale if necessary)

    Y_pred = model.predict(X_pred)

    if scaler:

        Y_pred = scaler.inverse_transform(Y_pred)     

    Y_pred = Y_pred[0,:,0]

    

    # skip the first few values (large errors due to initialization phase)

    Y_pred = Y_pred[skip_values:]

    Y_truth = Y_truth[skip_values:]

        

    # calculate errors

    abs_error = np.abs(Y_pred-Y_truth)

    mse_error = np.mean(abs_error**2)    

    

    return Y_pred, Y_truth, abs_error, mse_error



# This function outputs a plot showing the prediction vs. ground truth and the corresponding error.

def plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr=' '):



    fig, axes = plt.subplots(1, 2, sharex = True, figsize=(15, 5))



    # Temperature values

    axes[0].plot(Y_truth, 'r')

    axes[0].plot(Y_pred, 'b')

    axes[0].set_title('Prediction vs. ground truth')

    axes[0].legend(['Truth', 'Prediction'])

    axes[0].set_xlabel('sample')

    axes[0].set_ylabel('Temperature')

    

    # Error

    axes[1].plot(abs_error, 'r')

    axes[1].set_title('Error (total MSE: {:.5f})'.format(mse_error))

    axes[1].set_xlabel('sample')

    axes[1].set_ylabel('Error')

    

    fig.suptitle(descr, fontsize="x-large", fontweight='bold')



    return



# This function finds the load cycle with the highest mse.

def get_worst_cycle(model, X, Y, weights):

    

    highest_mse = 0

    worst_pid = 0



    for pid in np.arange(0,X.shape[0]):

        X_pred = X[pid:pid+1,:,:]

        Y_truth = Y[pid:pid+1,:,:]

        X_weights = weights[pid:pid+1,:]

        Y_pred, Y_truth, abs_error, mse_error = eval_model(model, X_pred, Y_truth, X_weights)

        if mse_error > highest_mse:

            highest_mse = mse_error

            worst_pid = pid

    

    return worst_pid
# Plot prediction vs. ground truth for one sample of training set



# training set

pid = 49 # basically a random load cycle

X_pred_train = X_train[pid:pid+1,:,:]

Y_truth_train = Y_train[pid:pid+1,:,:]

weights_train = X_train_weights[pid:pid+1,:]

Y_pred, Y_truth, abs_error, mse_error = eval_model(simple_lstm, X_pred_train, Y_truth_train, weights_train)

plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr='simple_LSTM - TRAIN SET EXAMPLE')
# Plot worst load cycle from dev set (highest mse)



worst_pid = get_worst_cycle(simple_lstm, X_dev, Y_dev, X_dev_weights)



X_pred_dev = X_dev[worst_pid:worst_pid+1,:,:]

Y_truth_dev = Y_dev[worst_pid:worst_pid+1,:,:]

weights_dev = X_dev_weights[worst_pid:worst_pid+1,:]

Y_pred, Y_truth, abs_error, mse_error = eval_model(simple_lstm, X_pred_dev, Y_truth_dev, weights_dev)        

plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr='simple_LSTM - DEV SET - WORST MSE SAMPLE (Nr.: ' + str(pid) +')')
# Plot prediction vs. ground truth for one sample of training set



# training set

pid = 49 # basically a random load cycle

X_pred_train = X_train[pid:pid+1,:,:]

Y_truth_train = Y_train[pid:pid+1,:,:]

weights_train = X_train_weights[pid:pid+1,:]

Y_pred, Y_truth, abs_error, mse_error = eval_model(res_lstm, X_pred_train, Y_truth_train, weights_train)

plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr='res_LSTM - TRAIN SET EXAMPLE')
# Plot worst load cycle from dev set (highest mse)



worst_pid = get_worst_cycle(simple_lstm, X_dev, Y_dev, X_dev_weights)



X_pred_dev = X_dev[worst_pid:worst_pid+1,:,:]

Y_truth_dev = Y_dev[worst_pid:worst_pid+1,:,:]

weights_dev = X_dev_weights[worst_pid:worst_pid+1,:]

Y_pred, Y_truth, abs_error, mse_error = eval_model(res_lstm, X_pred_dev, Y_truth_dev, weights_dev)        

plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr='res_LSTM - DEV SET - WORST MSE SAMPLE (Nr.: ' + str(pid) +')')
# training set

pid = 49 # basically a random load cycle

X_pred_train = X_train[pid:pid+1,:,:]

Y_truth_train = Y_train[pid:pid+1,:,:]

weights_train = X_train_weights[pid:pid+1,:]

Y_pred, Y_truth, abs_error, mse_error = eval_model(TCN, X_pred_train, Y_truth_train, weights_train)

plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr='TCN - TRAIN SET EXAMPLE')
# Plot worst load cycle from dev set (highest mse)



worst_pid = get_worst_cycle(TCN, X_dev, Y_dev, X_dev_weights)



X_pred_dev = X_dev[worst_pid:worst_pid+1,:,:]

Y_truth_dev = Y_dev[worst_pid:worst_pid+1,:,:]

weights_dev = X_dev_weights[worst_pid:worst_pid+1,:]

Y_pred, Y_truth, abs_error, mse_error = eval_model(TCN, X_pred_dev, Y_truth_dev, weights_dev)        

plot_prediction(Y_pred, Y_truth, abs_error, mse_error, descr='TCN - DEV SET - WORST MSE SAMPLE (Nr.: ' + str(pid) +')')