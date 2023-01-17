from IPython.display import Image

Image("../input/imagefold/forecasting_image.jpg")
import os

import datetime



import IPython

import IPython.display

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import tensorflow as tf



mpl.rcParams['figure.figsize'] = (8, 6)

mpl.rcParams['axes.grid'] = False
zip_path = tf.keras.utils.get_file(

    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',

    fname='jena_climate_2009_2016.csv.zip',

    extract=True)

csv_path, _ = os.path.splitext(zip_path) #We load the dataset in a csv_file
df = pd.read_csv(csv_path) #let's read the csv file with pandas
def preprocessing(data):

    

    # Getting rid of outliers

    data.loc[df['wv (m/s)'] == -9999.0, 'wv (m/s)'] = 0.0

    data.loc[df['max. wv (m/s)'] == -9999.0, 'max. wv (m/s)'] = 0.0

    

    # Taking values every hours

    data = data[5::6]# df[start,stop,step]

    

    wv = data.pop('wv (m/s)')

    max_wv = data.pop('max. wv (m/s)')



    # Convert to radians.

    wd_rad = data.pop('wd (deg)')*np.pi / 180



    # Calculate the wind x and y components.

    data['Wx'] = wv*np.cos(wd_rad)

    data['Wy'] = wv*np.sin(wd_rad)



    # Calculate the max wind x and y components.

    data['max Wx'] = max_wv*np.cos(wd_rad)

    data['max Wy'] = max_wv*np.sin(wd_rad)

    

    date_time = pd.to_datetime(data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    timestamp_s = date_time.map(datetime.datetime.timestamp)

    

    day = 24*60*60 # Time is second within a single day

    year = 365.2425*day # Time in second withon a year



    data['Day sin'] = np.sin(timestamp_s * (2*np.pi / day))

    data['Day cos'] = np.cos(timestamp_s * (2*np.pi / day))

    data['Year sin'] = np.sin(timestamp_s * (2*np.pi / year))

    data['Year cos'] = np.cos(timestamp_s * (2*np.pi / year))

    

    return(data)
def split(data):

    

    n = data.shape[0]

    

    train_df = data.iloc[0: n * 70 //100] # "iloc" because we have to select the lines at the indicies 0 to int(n*0.7) compared to "loc"

    val_df = data.iloc[n * 70 //100 : n * 90 //100]

    test_df = data.iloc[n * 90 //100:]

    

    return(train_df, val_df, test_df)
df_processed = preprocessing(df)



train_df, val_df, test_df = split(df_processed)



train_mean = train_df.mean() # returns a one column panda dataframe (serie) containing the mean of every columns

train_std = train_df.std() # same with standard deviation



train_df = (train_df - train_mean)/train_std # As simple as that !

val_df = (val_df - train_mean)/train_std

test_df = (test_df - train_mean)/train_std
type(train_df) # Right now, data has a type Panda Dataframe, we'll need to turn it to numpy array.
lookback = 48 # Looking at all features for the past 2 days

delay = 24 # Trying to predict the temperature for the next day

window_length = lookback + delay

batch_size = 32 # Features will be batched 32 by 32.
def create_dataset(X, y, delay=24):

    # X and y should be pandas dataframes

    Xs, ys = [], []

    for i in range(lookback, len(X)-delay):

        v = X.iloc[i-lookback:i].to_numpy() # every one hour, we take the past 48 hours of features

        Xs.append(v)

        w = y.iloc[i+delay] # Every timestep, we take the temperature the next delay (here one day)

        ys.append(w)

    return(np.array(Xs), np.array(ys))
X_train, y_train = create_dataset(train_df, train_df['T (degC)'], delay = delay)

X_val, y_val = create_dataset(val_df, val_df['T (degC)'], delay = delay)
print("X_train shape is {}: ".format(X_train.shape))

print("y_train shape is {}: ".format(y_train.shape))



print("\nX_val shape is {}: ".format(X_val.shape))

print("y_val shape is {}: ".format(y_val.shape))
def naive_eval_arr(X, y, lookback, delay):

    batch_maes = []

    for i in range(0, len(X)):

        preds = X[i, -1, 1] #For all elements in the batch, we are saying the prediction of temperature is equal to the last temperature recorded within the 48 hours

        mae = np.mean(np.abs(preds - y[i]))

        batch_maes.append(mae)

    return(np.mean(batch_maes))



naive_loss_arr = naive_eval_arr(X_val, y_val, lookback = lookback, delay = delay)



naive_loss_arr = round(naive_eval_arr(X_val, y_val, lookback = lookback, delay = delay),2) # Round the value

print(naive_loss_arr)
from keras.models import Sequential

from keras.layers import Flatten, Dense

from keras.optimizers import RMSprop
# Let's start with a simple Dense model

model = Sequential([

    Flatten(input_shape=(lookback, 19)),

    Dense(32, activation='relu'),

    Dense(1) # We try to predict only one value for now

])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_val, y_val), batch_size = 32)
# Let's define a function to plot graphs, it will be usefull since we'll built a lot of them !

def plot(history, naive_loss):

    

    loss = history.history['loss']

    val_loss = history.history['val_loss']



    epochs = range(1, len(loss)+1)

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.axhline(y=naive_loss, color ='r')



    plt.legend()

    plt.show()
plot(history, naive_loss_arr)
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=32, step=1):

    if max_index is None:

        max_index = len(data) - delay - 1

    i = min_index + lookback

    while True:

        if shuffle == True: # pay attention ! We are not shuffeling timesteps but elemnts within a batch ! It is important to keep the data in time order

            rows = np.random.randint(min_index + lookback, max_index-delay-1, size=batch_size) # return an array containing size elements ranging from min_index+lookback to max_index

        else:

            if i + batch_size >= max_index-delay-1: #Since we are incrementing on "i". If its value is greater than the max_index --> start from the begining

                i = min_index + lookback # We need to start from the indice lookback, since we want to take lookback elements here.

            rows = np.arange(i, min(i + batch_size, max_index)) # Just creating an array that contain the indices of each sample in the batch

            i+=len(rows) # rows represents the number of sample in one batch

            

        samples = np.zeros((len(rows), lookback//step, data.shape[-1])) # shape = (batch_size, lookback, nbr_of_features)

        targets = np.zeros((len(rows),)) #Shape = (batch_size,)

        

        for j, row in enumerate(rows):

            indices = range(rows[j] - lookback, rows[j], step) #From one indice given by rows[j], we are picking loockback previous elements in the dataset

            samples[j] = data[indices]

            targets[j] = data[rows[j] + delay][1] #We only want to predict the temperature for now,since [1], the second column

        yield samples, targets # The yield that replace the return to create a generator and not a regular function.
data_train = train_df.to_numpy()

train_gen = generator(data = data_train, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_train), shuffle = True, batch_size = batch_size)



data_val = val_df.to_numpy()

val_gen = generator(data = data_val, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_val), batch_size = batch_size)



data_test = test_df.to_numpy()

test_gen = generator(data = data_val, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_test), batch_size = batch_size)
train_gen # It is a generator
print(next(iter(train_gen))[0].shape) # Here we picked one batch of samples
print(next(iter(train_gen))[1].shape)
def naive_eval_gen(generator):

    batch_maes = []

    for step in range(len(data_val)-lookback):

        samples, targets = next(generator)

        preds = samples[:,-1,1] #For all elements in the batch, we are saying the prediction of temperature is equal to the last temperature recorded within the 48 hours

        mae = np.mean(np.abs(preds - targets))

        batch_maes.append(mae)

    return(np.mean(batch_maes))



naive_loss_gen = round(naive_eval_gen(val_gen),2)

print(naive_loss_gen)
model = Sequential([

    Flatten(input_shape=(lookback, 19)),

    Dense(32, activation='relu'),

    Dense(1)

])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history = model.fit_generator(train_gen, epochs = 30, steps_per_epoch = data_train.shape[0]//batch_size, validation_data = val_gen, validation_steps = data_val.shape[0]//batch_size)
plot(history, naive_loss_gen)
train_df_it = tf.data.Dataset.from_tensor_slices(data_train) # *from_tensor_slices* is the function to call to turn numpy array to tf.data.Dataset
train_df_it # It is a Dataset that contains elements of shape (19,)
for element in train_df_it:

    print(element)

    break # Use break if you don't want to see the whole dataset
for element in train_df_it.as_numpy_iterator():

    print(element)

    break
train_it = train_df_it.window(size = window_length, shift = 1, drop_remainder = True)

train_it = train_it.flat_map(lambda window: window.batch(window_length))

train_it = train_it.map(lambda windows: (windows[:48,:], windows[-1,1]))
train_df_it = tf.data.Dataset.from_tensor_slices(data_train)

val_df_it = tf.data.Dataset.from_tensor_slices(data_val)





# window(size, shift=None, stride=1, drop_remainder=False)

train_it = train_df_it.window(size = window_length, shift = 1, drop_remainder = True)

train_it = train_it.flat_map(lambda window: window.batch(window_length))

train_it = train_it.shuffle(buffer_size = df.shape[0]).batch(batch_size)

train_it = train_it.map(lambda windows: (windows[:,:48,:], windows[:,-1,1]))



#train_it = train_it.map(lambda windows: (windows[:,:48,:], windows[:,-1,1]))





val_it = val_df_it.window(window_length, shift = 1, drop_remainder = True)

val_it = val_it.flat_map(lambda window: window.batch(window_length))

val_it = val_it.shuffle(buffer_size = df.shape[0]).batch(batch_size)

val_it = val_it.map(lambda windows: (windows[:,:48,:], windows[:,-1,1]))



#val_it = val_it.map(lambda windows: (windows[:,:48,:], windows[:,-1,1]))

def naive_eval_it(iterator):

    batch_maes = []

    for tupple in iterator.as_numpy_iterator():

        samples, targets = tupple[0], tupple[1]

        for sample, target in zip(samples, targets):

            preds = sample[-1,1]

            mae = np.mean(np.abs(preds - target))

            batch_maes.append(mae)

    return(np.mean(batch_maes))



naive_loss_it = round(naive_eval_it(val_it),2)

print(naive_loss_it)
model = Sequential([

    Flatten(input_shape=(lookback, 19)),

    Dense(32, activation='relu'),

    Dense(1)

])
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

history = model.fit(train_it, epochs = 30, validation_data = val_it)
plot(history, naive_loss_it)
lookback = 48

delay = 24

window_length = lookback + delay

batch_size = 32

buffer_size = len(train_df)
# Still using these:

print("train_df shape is: {}".format(train_df.shape))

print("val_df shape is: {}".format(val_df.shape))

print("test_df shape is: {}".format(test_df.shape))
def create_arr_dataset(data, lookback, delay):

    data = data.to_numpy()

    X, y = [], []

    for i in range(lookback, len(data)-delay):

        X.append(data[i-lookback: i,:])

        y.append(data[i:i+delay, 1])

    X = np.array(X)

    y = np.array(y)

    return(X, y)
X_train, y_train = create_arr_dataset(train_df, lookback = lookback, delay = delay)

X_val, y_val = create_arr_dataset(val_df, lookback = lookback, delay = delay)
print("X_train shape is: {}, and y_train shape is: {}".format(X_train.shape, y_train.shape))

print("X_val shape is: {}, and y_val shape is: {}".format(X_val.shape, y_val.shape))
def naive_loss_arr2(X, y, lookback, delay):

    batch_maes, batch_rmses = [],  []

    for sample, target in zip(X, y):

        preds = []

        for i in range(delay):

            preds.append(sample[-1,1]) # For the next 8 hours, we predict the value to be the same as the one in the last timestep of the sample

        preds = np.array(preds)

        mae = np.mean(np.abs(preds - target))

        rmse = np.sqrt(np.mean((preds - target)**2))

        batch_maes.append(mae)

        batch_rmses.append(rmse)

    return(np.mean(batch_maes), np.mean(batch_rmses))



naive_loss_arr2_mae, naive_loss_arr2_rmse = naive_loss_arr2(X_val, y_val, lookback = lookback, delay= delay)



naive_loss_arr2_mae = round(naive_loss_arr2_mae,2)

naive_loss_arr2_rmse = round(naive_loss_arr2_rmse,2)



print("mae is: {}, rmse is: {}".format(naive_loss_arr2_mae, naive_loss_arr2_rmse))
model = Sequential([

    Flatten(input_shape=(lookback, 19)),

    Dense(32, activation='relu'),

    Dense(delay) # Pay attention, now we want to predict the temperature for the next 8 hours

])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_val, y_val))
plot(history, naive_loss_arr2_mae)
def generator_2(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=32, step=1):

    if max_index is None:

        max_index = len(data) - delay - 1

    i = min_index + lookback

    while True:

        if shuffle == True: # pay attention ! We are not shuffeling timesteps but elemnts within a batch ! It is important to keep the data in time order

            rows = np.random.randint(min_index + lookback, max_index-delay-1, size=batch_size) # return an array containing size elements ranging from min_index+lookback to max_index

        else:

            if i + batch_size >= max_index-delay-1: #Since we are incrementing on "i". If its value is greater than the max_index --> start from the begining

                i = min_index + lookback # We need to start from the indice lookback, since we want to take lookback elements here.

            rows = np.arange(i, min(i + batch_size, max_index)) # Just creating an array that contain the indices of each sample in the batch

            i+=len(rows) # rows represents the number of sample in one batch

            

        samples = np.zeros((len(rows), lookback//step, data.shape[-1])) # shape = (batch_size, lookback, nbr_of_features)

        targets = np.zeros((len(rows),delay)) #Shape = (batch_size,delay)

        

        for j, row in enumerate(rows): # We loop here for batch_size ie 32 loops

            indice_samples = range(rows[j] - lookback, rows[j], step) #From one indice given by rows[j], we are picking loockback previous elements in the dataset

            indice_targets = range(rows[j], rows[j]+delay, step)

            samples[j] = data[indice_samples]

            targets[j] = data[:,1][indice_targets] #We only want to predict the temperature for now,since [1], the second column

        yield samples, targets # The yield that replace the return to create a generator and not a regular function.
data_train = train_df.to_numpy()

train_gen = generator_2(data = data_train, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_train), shuffle = True, batch_size = batch_size)



data_val = val_df.to_numpy()

val_gen = generator_2(data = data_val, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_val), batch_size = batch_size)



data_test = test_df.to_numpy()

test_gen = generator_2(data = data_val, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_test), batch_size = batch_size)
train_gen
def naive_loss_gen2(generator):

    batch_maes, batch_rmses = [],  []



    for batch_number in range(0, 500):

        batch_sample, batch_target = next(generator)



        preds = np.zeros((batch_size, delay))

        for j, sample in enumerate(batch_sample):

            for i in range(0, delay):

                preds[j,i] = sample[-1,1]



            mae = np.mean(np.abs(preds[j] - batch_target))

            rmse = np.sqrt(np.mean((preds[j] - batch_target)**2))

    

            batch_maes.append(mae)

            batch_rmses.append(rmse)

    

    return(np.mean(batch_maes), np.mean(batch_rmses))



naive_loss_gen2_mae, naive_loss_gen2_rmse = naive_loss_gen2(val_gen)



naive_loss_gen2_mae = round(naive_loss_gen2_mae,2)

naive_loss_gen2_rmse = round(naive_loss_gen2_rmse,2)



print("mae is: {}, rmse is: {}".format(naive_loss_gen2_mae, naive_loss_gen2_rmse))
model = Sequential([

    Flatten(input_shape=(lookback, 19)),

    Dense(32, activation='relu'),

    Dense(delay)

])
train_gen2 = generator_2(data = data_train, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_train), shuffle = True, batch_size = batch_size)

val_gen2 = generator_2(data = data_val, lookback = lookback, delay =delay, min_index = 0, max_index = len(data_val), batch_size = batch_size)



model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history = model.fit_generator(train_gen2, epochs = 30, steps_per_epoch = data_train.shape[0]//batch_size, validation_data = val_gen2, validation_steps = data_val.shape[0]//batch_size)
plot(history, naive_loss_gen2_mae)
data_train = train_df.to_numpy()

data_val = val_df.to_numpy()

data_test = test_df.to_numpy()



train_df_it = tf.data.Dataset.from_tensor_slices(data_train)

val_df_it = tf.data.Dataset.from_tensor_slices(data_val)

test_df_it = tf.data.Dataset.from_tensor_slices(data_test)



# window(size, shift=None, stride=1, drop_remainder=False)

train_it = train_df_it.window(size = window_length, shift = 1, drop_remainder = True)

train_it = train_it.flat_map(lambda window: window.batch(window_length))

train_it = train_it.shuffle(buffer_size = df.shape[0]).batch(batch_size)

train_it = train_it.map(lambda windows: (windows[:,:lookback,:], windows[:,lookback:lookback+delay,1]))





val_it = val_df_it.window(window_length, shift = 1, drop_remainder = True)

val_it = val_it.flat_map(lambda window: window.batch(window_length))

val_it = val_it.shuffle(buffer_size = df.shape[0]).batch(batch_size)

val_it = val_it.map(lambda windows: (windows[:,:lookback,:], windows[:,lookback:lookback+delay,1]))



test_it = test_df_it.window(window_length, shift = 1, drop_remainder = True)

test_it = test_it.flat_map(lambda window: window.batch(window_length))

test_it = test_it.shuffle(buffer_size = df.shape[0]).batch(batch_size)

test_it = test_it.map(lambda windows: (windows[:,:lookback,:], windows[:,lookback:lookback+delay,1]))
print(next(iter(val_it))[0].shape)

print(next(iter(val_it))[1].shape)
def naive_eval_it2(iterator):

    batch_maes, batch_rmses = [],  []

    for tupple in iterator.as_numpy_iterator():

        samples, targets = tupple[0], tupple[1]

        for sample, target in zip(samples, targets):

            preds = []

            for i in range(0,delay):

                preds.append(sample[-1,1])

            preds = np.array(preds)

            

            mae = np.mean(np.abs(preds - target))

            rmse = np.sqrt(np.mean((preds - target)**2))

    

            batch_maes.append(mae)

            batch_rmses.append(rmse)

    return(np.mean(batch_maes), np.mean(batch_rmses))



naive_loss_it2_mae, naive_loss_it2_rmse = naive_eval_it2(val_it)



naive_loss_it2_mae = round(naive_loss_it2_mae,2)

naive_loss_it2_rmse = round(naive_loss_it2_rmse,2)



print("mae is: {}, rmse is: {}".format(naive_loss_it2_mae, naive_loss_it2_rmse))
model_linear = Sequential([

    Flatten(input_shape=(lookback, 19)),

    Dense(32, activation='relu'),

    Dense(delay)

])
model_linear.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history_linear = model_linear.fit(train_it, epochs = 30, validation_data = val_it)
plot(history_linear, naive_loss_it2_mae)
from keras.layers import LSTM, GRU, SimpleRNN
model_LSTM = Sequential([

    LSTM(30, return_sequences = True, input_shape = [None, 19], dropout=0.2, recurrent_dropout = 0.2),

    LSTM(30, dropout=0.2, recurrent_dropout = 0.2),

    Dense(delay)

])
model_LSTM.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history_LSTM = model_LSTM.fit(train_it, epochs = 30, validation_data = val_it)
plot(history_LSTM, naive_loss_it2_mae)
tupple_test = test_it.as_numpy_iterator()

tupple_test = next(tupple_test)

batch_sample_test, batch_target_test = tupple_test 

for sample_test, target_test in zip(batch_sample_test, batch_target_test):

    

    # Arvesting data for the reals points

    sample_points, target_points = sample_test[:,1], target_test

    

    # Arvesting data for the naive model

    naive_points = np.zeros(delay)

    for i in range(delay):

        naive_points[i] = sample_test[-1,1]

    

    # Arvesting data for the linear evaluation

    ypreds_linear = model_linear.predict(batch_sample_test)

    linear_points = ypreds_linear[0,:]

    

    # Arvesting data for the LSTM evaluation

    ypreds_LSTM = model_LSTM.predict(batch_sample_test)

    LSTM_points = ypreds_LSTM[0,:]

    break
real_points = np.concatenate((sample_points, target_points), axis=None)

naive_points = np.concatenate((sample_points, naive_points), axis=None)

linear_points = np.concatenate((sample_points, linear_points), axis=None)

LSTM_points = np.concatenate((sample_points, LSTM_points), axis=None)
epochs = range(1, window_length+1)

plt.figure()

plt.plot(epochs, real_points, 'bo', label='Real data')

plt.plot(epochs, naive_points, 'b', label='Naive evaluation')

plt.plot(epochs, linear_points, 'r', label='Linear evaluation')

plt.plot(epochs, LSTM_points, 'g', label='LSTM evaluation')

plt.title('Compararing the evaluation from different models')

#plt.axhline(y=naive_loss, color ='r')



plt.legend()

plt.show()
def create_arr_dataset(data, lookback, delay):

    data = data.to_numpy()

    X = np.zeros(data.shape[0], lookback, data.shape[1])

    y = np.zeros(data.shape[0], lookback, delay)

    for i in range(lookback, len(data)-delay):

        X[i] = data[i-lookback:i,:]

        for j in range(lookback):

            y[i,j] = data[i-lookback + j : i + j + delay,1]

    return(X, y)
from keras.layers import MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dense
model_CNN = Sequential([

    Conv1D(filters = 32, kernel_size = 5, activation='relu', input_shape = (None, window_length)),

    MaxPooling1D(3),

    Conv1D(filters = 32, kernel_size = 5, activation='relu'),

    GlobalMaxPooling1D(),

    Dense(delay)

])
model_CNN.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')

history_CNN = model.fit(train_it, epochs = 30, validation_data = val_it)
loss = history_CNN.history['loss']

val_loss = history_CNN.history['val_loss']



epochs = range(1, len(loss)+1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')



plt.legend()

plt.show()