import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

import pickle

from pathlib import Path

import datetime

import time

import itertools



import tensorflow as tf

from tensorflow.keras.models import Model, load_model

from tensorflow.keras import layers as KL

from sklearn.preprocessing import MinMaxScaler



# Check GPUs:",

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            # Prevent TensorFlow from allocating all memory of all GPUs:

            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:

        print(e)
# Kaggle paths (uncomment to use):

df = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

df_mob = pd.read_csv('../input/covid19-mobility-data/Global_Mobility_Report.csv')



# Client paths (uncomment to use):

#df = pd.read_csv('data/train.csv')

#df_mob = pd.read_csv('data/Global_Mobility_Report.csv')



DAYS_IN_DF = int(len(df[df['Country_Region'] == 'Afghanistan'])/2)

START_DATE = datetime.datetime(2020,2,15) # Including

CUTOFF_DATE = datetime.datetime(2020,5,25) # Up to and including

TRAIN_TEST_SPLIT_DATE = datetime.datetime(2020,4,26) # Up to and including

DAYS = (CUTOFF_DATE - START_DATE).days + 1 # +1 because of including start and end

print(DAYS)



VARIABLES = ['ConfirmedCases', 'Fatalities']

N_FEATURES_OUT = len(VARIABLES)

# N_FEATURES_IN derived later in the notebook

df = df[df['Target'].isin(VARIABLES)]



# Set country

# Be aware that many countries are divided into sub-regions. The code only works out-of-the-box

# with smaller countries that are represented in the dataframe as a single geographical entity.

# It is however easily modified by aggregating time series

COUNTRY = 'Sweden'



N_STEPS_IN, N_STEPS_OUT = 15, 10



df
def get_country_mobility(country_name):

    df_mob_country = df_mob[(df_mob['country_region'] == COUNTRY) & (df_mob['sub_region_1'].isnull())].iloc[:,5:]

    print(df_mob_country.shape)

    if df_mob_country.isnull().values.any():

        print(df_mob_country.isnull().values.sum(), 'NaNs detected. Interpolating to get rid of them.')

        df_mob_country.interpolate(method='linear', axis=0, inplace = True)

    assert not df_mob_country.isnull().values.any() and df_mob_country.shape[0] == DAYS

    return df_mob_country.to_numpy()



data_mob_country = get_country_mobility(COUNTRY)

data_mob_country[:5,:]
plt.plot(data_mob_country)
# Function needed to go back and forth between dates and indices in numpy arrays

def get_int_idx_from_date(df, date):

    idx = df[(df['Date'] == date.strftime('%Y-%m-%d')) & 

             (df['Country_Region'] == 'Afghanistan')].index.to_list()

    return idx[0]



print(get_int_idx_from_date(df, START_DATE), get_int_idx_from_date(df, TRAIN_TEST_SPLIT_DATE))



def df_to_npy(df, start_date, end_date, country_name = None):

    # Date slicing

    df.loc[:,'Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Setting negative values to 0

    df.loc[:,'TargetValue'] = np.clip(df["TargetValue"], 0, None)

    

    # From long to wide format

    df.loc[:,'new_id'] = np.repeat(np.arange(int(len(df)/2)), 2)

    df_new = df.pivot(index = 'new_id', columns = 'Target', values = 'TargetValue')

    df = df[df['Target'] == 'ConfirmedCases']

    df = df.drop(columns = ['Target', 'TargetValue'])

    df.set_index('new_id', inplace = True)

    df = df.merge(df_new, left_index = True, right_index = True)

    df_new = df

    data = df.loc[:,['ConfirmedCases', 'Fatalities']].to_numpy()

    

    l = []

    for i in range(int(len(data)/DAYS)):

        l.append(data[i*DAYS:(i+1)*DAYS, :])

    data = np.concatenate(l, axis = 1)



    return data, df_new



data, df_new = df_to_npy(df, START_DATE, CUTOFF_DATE)

print(data.shape)



def get_int_idx_from_country_region(df, data, country_region):

    days = data.shape[0]

    idx = df[df['Country_Region'] == country_region].index[0]

    idx_confirmedcases = idx / (days/2)

    assert idx_confirmedcases.is_integer()

    idx_confirmedcases = int(idx_confirmedcases)

    idx_fatalities = idx_confirmedcases + 1

    return idx_confirmedcases, idx_fatalities



# Sanity check

print(get_int_idx_from_country_region(df_new, data, 'Afghanistan'), 

      get_int_idx_from_country_region(df_new, data, 'Albania'))
plt.plot(data[:,get_int_idx_from_country_region(df_new, data, 'Sweden')])
if COUNTRY != 'ALL':

    data = data[:,get_int_idx_from_country_region(df_new, data, COUNTRY)]

    

print(data.shape)
def combine_datasets(data_1, data_2):

    print(data_1.shape, data_2.shape)

    data = np.concatenate((data_1, data_2), axis = 1)

    print(data.shape)

    return data

data = combine_datasets(data, data_mob_country)
def train_test_split(df, data, split_date):

    split_idx = get_int_idx_from_date(df, split_date)

    train = data[:split_idx,:]

    test = data[split_idx:,:]

    return train, test



train, test = train_test_split(df_new, data, TRAIN_TEST_SPLIT_DATE)

print(train.shape, test.shape)



N_FEATURES_IN = data.shape[1]
# split a multivariate sequence into samples

def split_sequences(sequences, n_steps_in, n_steps_out):

    X, y = list(), list()

    for i in range(len(sequences)):

        # find the end of this pattern

        end_ix = i + n_steps_in

        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the dataset

        if out_end_ix > len(sequences):

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)



X_train, Y_train = split_sequences(train, N_STEPS_IN, N_STEPS_OUT)

Y_train = Y_train[:,:,:N_FEATURES_OUT]

print(X_train.shape, Y_train.shape)
def split_sequences_train_test_overlap(train, test, n_steps_in, n_steps_out):

    X_val, Y_val = list(), list()

    X_test, Y_test = list(), list()

    for i in range(len(test)):

        # find the end of this pattern

        train_start_ix = len(train) - n_steps_in + i

        #test_end_ix = i + n_steps_in

        test_out_end_ix = i + n_steps_out

        # check if we are beyond the dataset

        #print(i, train_start_ix, test_out_end_ix)

        if test_out_end_ix > len(test):

            break

        # gather input and output parts of the pattern

        seq_x = np.concatenate((train[train_start_ix:, :], test[:i, :]))

        seq_y = test[i:test_out_end_ix, :]

        #print(seq_x.shape)

        #print(seq_y.shape)

        if train_start_ix >= len(train):

            #print(train_start_ix)

            #print(train[train_start_ix:, :].shape)

            seq_x = test[i-n_steps_in:i, :]

            seq_y = test[i:test_out_end_ix, :]

            #print(seq_x.shape, seq_y.shape)

            X_test.append(seq_x)

            Y_test.append(seq_y)

        else:

            X_val.append(seq_x)

            Y_val.append(seq_y)

    return np.array(X_val), np.array(Y_val), np.array(X_test), np.array(Y_test)



X_val, Y_val, X_test, Y_test = split_sequences_train_test_overlap(train, test, N_STEPS_IN, N_STEPS_OUT)

Y_val = Y_val[:,:,:N_FEATURES_OUT]

Y_test = Y_test[:,:,:N_FEATURES_OUT]



print(X_val.shape, Y_val.shape)

print(X_test.shape, Y_test.shape)
X_scaler = MinMaxScaler((0,1))

Y_scaler = MinMaxScaler((0,1))

def scale(arr, x_or_y, fit = False):

    arr_shape = arr.shape

    arr = arr.reshape(-1, arr_shape[2])

    if x_or_y == 'x':

        scaler = X_scaler

    elif x_or_y == 'y':

        scaler = Y_scaler

    if fit:

        scaler.fit(arr)

    arr = scaler.transform(arr)

    arr = arr.reshape(arr_shape)

    return arr



X_train = scale(X_train, 'x', fit = True)

X_val = scale(X_val, 'x')

X_test = scale(X_test, 'x')



Y_train = scale(Y_train, 'y', fit = True)

Y_val = scale(Y_val, 'y')

Y_test = scale(Y_test, 'y')
def unscale(arr, x_or_y):

    arr_shape = arr.shape

    arr = arr.reshape(-1, arr_shape[2])

    if x_or_y == 'x':

        scaler = X_scaler

    elif x_or_y == 'y':

        scaler = Y_scaler

    arr = scaler.inverse_transform(arr)

    arr = arr.reshape(arr_shape)

    return arr
def get_cnn_lstm_model(n_steps_in, n_features_in, n_steps_out, n_features_out, 

                       dropout = 0, conv_filter_sizes = (3,3), LSTM_size = 200, 

                       dense_size_scaler = 10):

    seq_inp = KL.Input(shape=(n_steps_in, n_features_in))

    

    x = KL.Conv1D(filters=64, kernel_size=conv_filter_sizes[0], activation='relu')(seq_inp)

    x = KL.Conv1D(filters=64, kernel_size=conv_filter_sizes[1], activation='relu')(x)

    x = KL.MaxPooling1D(pool_size=2)(x)

    x = KL.Flatten()(x)

    x = KL.RepeatVector(n_steps_out)(x)

    x = KL.LSTM(LSTM_size, activation='relu', return_sequences=True, dropout = dropout)(x)

    x = KL.Dropout(dropout)(x)

    x = KL.TimeDistributed(KL.Dense(int(n_features_out*dense_size_scaler), activation = 'relu'))(x)

    x = KL.Dropout(dropout)(x)

    x = KL.TimeDistributed(KL.Dense(n_features_out))(x)

    

    model = Model(inputs = seq_inp, outputs = x)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

    model.compile(loss='mse', optimizer=optimizer)

    

    return model



get_cnn_lstm_model(N_STEPS_IN, N_FEATURES_IN, N_STEPS_OUT, N_FEATURES_OUT, 

                   dropout = 0.2, conv_filter_sizes = (3,3), LSTM_size = 200, 

                   dense_size_scaler = 10).summary()
def save_history(hist, tag=''):

    Path('histories').mkdir(exist_ok=True)

    with open('histories/history_' + tag + '.pickle', 'wb') as dumpfile:

        pickle.dump(hist.history, dumpfile)



def load_history(tag):

    with open('histories/history_' + tag + '.pickle', 'rb') as f:

        return pickle.load(f)

    

def load_history_kaggle(tag):

    with open('../input/covid19-forecasting-models-histories/histories/history_' + tag + '.pickle', 'rb') as f:

        return pickle.load(f)
print(X_train.shape, Y_train.shape)

print(X_val.shape, Y_val.shape)

print(X_test.shape, Y_test.shape)



# Hyper-parameters to be tuned:

DROPOUTS = [0, 0.25, 0.5]

CONV_FILTER_SIZES = [(3,3), (5,5), (5,3)]

LSTM_SIZES = [100, 200, 300]

DENSE_SIZE_SCALER = [10, 100, 1000]



BATCH_SIZES = [8, 16, X_train.shape[0]]



EPOCHS = 500



histories = []



i = 0

for batch_size in BATCH_SIZES:

    for dropout in DROPOUTS:

        for conv_filter_size in CONV_FILTER_SIZES:

            for lstm_size in LSTM_SIZES:

                for dense_size_scaler in DENSE_SIZE_SCALER:

                    t0 = time.time()

                    folder = 'models/'

                    filename = str('sweden_cnn_lstm-' + str(i).zfill(3) + 

                                   '_bsize-' + str(batch_size) +

                                   '_drop-' + str(dropout) + 

                                   '_conv_sz-' + str(conv_filter_size) + 

                                   '_lstm_sz-' + str(lstm_size) +

                                   '_dense_sz' + str(dense_size_scaler))

                    #print('Training model', filename)

                    

                    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

                        filepath = folder + filename + '_epoch-{epoch:02d}_loss-{val_loss:.5f}.h5',

                        save_weights_only=False,

                        monitor='val_loss',

                        mode='min',

                        save_best_only=True)



                    #model = get_cnn_lstm_model(N_STEPS_IN, N_FEATURES_IN, N_STEPS_OUT, N_FEATURES_OUT, 

                    #                           dropout = dropout, conv_filter_sizes = conv_filter_size, 

                    #                           LSTM_size = lstm_size, dense_size_scaler = dense_size_scaler)

                    #history = model.fit(X_train, Y_train, 

                    #                    validation_data = (X_val, Y_val), 

                    #                    epochs=EPOCHS, batch_size = batch_size, verbose=0, 

                    #                    callbacks=[model_checkpoint_callback], 

                    #                    shuffle = True)

                    #histories.append(history)

                    #save_history(history, filename)

                    

                    t1 = time.time()

                    total = t1-t0

                    #print(str(EPOCHS), 'epochs of training finished in', 

                    #      '{0:.2f}'.format(total), 'seconds')

                    i += 1
def plot_history(history, ylim = None, title = None):

    fig, ax = plt.subplots(1, 1, figsize = (10,5))

    ax.plot(history['loss'])

    ax.plot(history['val_loss'])

    ax.set_title(title)

    ax.set_ylabel('loss')

    ax.set_xlabel('epoch')

    ax.set_ylim(ylim)

    ax.legend(['train', 'val'], loc='upper left')

    

def get_k_best_models(models_dir_path, k):

    p = Path(models_dir_path)

    l = list(p.glob('*.h5'))

    

    # Extracting the val_loss value in the file name

    names = [path.stem for path in l]

    losses = [float(loss) for loss in [path.stem[-7:] for path in l]]

    d = {}

    for i in range(len(names)):

        d[names[i]] = losses[i]

    # Sort the dict

    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}

    # Return the lowest k val losses

    return dict(itertools.islice(d.items(), k))



#best10 = get_k_best_models('models', 10)



# Client path (uncomment to use):

#path_to_best10_pickle = 'models/best10_models_dict.pickle'



# Kaggle path (uncomment to use):

path_to_best10_pickle = '../input/covid19-forecasting-models-histories/best10_models_dict.pickle'



#with open(path_to_best10_pickle, 'wb') as f:

#    pickle.dump(best10, f)



with open(path_to_best10_pickle, 'rb') as f:

    best10 = pickle.load(f)



for name, val_loss in best10.items():

    print(name, val_loss)
for name in best10.keys():

    name = name[:-23] #-23 to remove epoch and loss substrings

    name = name.replace(',', '') # Needed on Kaggle

    #history = load_history(name)

    history = load_history_kaggle(name)

    plot_history(history, ylim = (0,0.1), title = name)
# Model 054

EPOCHS = 3000

batch_size = 8

dropout = 0.5

conv_filter_size = (3,3)

lstm_size = 100

dense_size_scaler = 10



t0 = time.time()

folder = 'models/longruns/'

filename = str('sweden_cnn_lstm-' + '054' + 

               '_bsize-' + str(batch_size) +

               '_drop-' + str(dropout) + 

               '_conv_sz-' + str(conv_filter_size) + 

               '_lstm_sz-' + str(lstm_size) +

               '_dense_sz' + str(dense_size_scaler))

print('Training model', filename)



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath = folder + filename + '_epoch-{epoch:02d}_loss-{val_loss:.5f}.h5',

    save_weights_only=False,

    monitor='val_loss',

    mode='min',

    save_best_only=True)



model = get_cnn_lstm_model(N_STEPS_IN, N_FEATURES_IN, N_STEPS_OUT, N_FEATURES_OUT, 

                           dropout = dropout, conv_filter_sizes = conv_filter_size, 

                           LSTM_size = lstm_size, dense_size_scaler = dense_size_scaler)

model.summary()

#history = model.fit(X_train, Y_train, 

#                    validation_data = (X_val, Y_val), 

#                    epochs=EPOCHS, batch_size = batch_size, verbose=0, 

#                    callbacks=[model_checkpoint_callback], 

#                    shuffle = True)

#save_history(history, str('longrun-' + filename))



t1 = time.time()

total = t1-t0

print(str(EPOCHS), 'epochs of training finished in', 

      '{0:.2f}'.format(total), 'seconds')
filename = filename.replace(',', '') # Needed on Kaggle

history_054_lr = load_history_kaggle(str('longrun-' + filename))

plot_history(history_054_lr, ylim = (0,0.1))
# Model 066

EPOCHS = 3000

batch_size = 8

dropout = 0.5

conv_filter_size = (5,5)

lstm_size = 200

dense_size_scaler = 10



t0 = time.time()

folder = 'models/longruns/'

filename = str('sweden_cnn_lstm-' + '066' + 

               '_bsize-' + str(batch_size) +

               '_drop-' + str(dropout) + 

               '_conv_sz-' + str(conv_filter_size) + 

               '_lstm_sz-' + str(lstm_size) +

               '_dense_sz' + str(dense_size_scaler))

print('Training model', filename)



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath = folder + filename + '_epoch-{epoch:02d}_loss-{val_loss:.5f}.h5',

    save_weights_only=False,

    monitor='val_loss',

    mode='min',

    save_best_only=True)



model = get_cnn_lstm_model(N_STEPS_IN, N_FEATURES_IN, N_STEPS_OUT, N_FEATURES_OUT, 

                           dropout = dropout, conv_filter_sizes = conv_filter_size, 

                           LSTM_size = lstm_size, dense_size_scaler = dense_size_scaler)

model.summary()

#history = model.fit(X_train, Y_train, 

#                    validation_data = (X_val, Y_val), 

#                    epochs=EPOCHS, batch_size = batch_size, verbose=0, 

#                    callbacks=[model_checkpoint_callback], 

#                    shuffle = True)

#save_history(history, str('longrun-' + filename))



t1 = time.time()

total = t1-t0

print(str(EPOCHS), 'epochs of training finished in', 

      '{0:.2f}'.format(total), 'seconds')
filename = filename.replace(',', '') # Needed on Kaggle

history_066_lr = load_history_kaggle(str('longrun-' + filename))

plot_history(history_066_lr, ylim = (0,0.1))
# Model 234

EPOCHS = 3000

batch_size = 47

dropout = 0.5

conv_filter_size = (5,3)

lstm_size = 100

dense_size_scaler = 10



t0 = time.time()

folder = 'models/longruns/'

filename = str('sweden_cnn_lstm-' + '234' + 

               '_bsize-' + str(batch_size) +

               '_drop-' + str(dropout) + 

               '_conv_sz-' + str(conv_filter_size) + 

               '_lstm_sz-' + str(lstm_size) +

               '_dense_sz' + str(dense_size_scaler))

print('Training model', filename)



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(

    filepath = folder + filename + '_epoch-{epoch:02d}_loss-{val_loss:.5f}.h5',

    save_weights_only=False,

    monitor='val_loss',

    mode='min',

    save_best_only=True)



model = get_cnn_lstm_model(N_STEPS_IN, N_FEATURES_IN, N_STEPS_OUT, N_FEATURES_OUT, 

                           dropout = dropout, conv_filter_sizes = conv_filter_size, 

                           LSTM_size = lstm_size, dense_size_scaler = dense_size_scaler)

model.summary()

#history = model.fit(X_train, Y_train, 

#                    validation_data = (X_val, Y_val), 

#                    epochs=EPOCHS, batch_size = batch_size, verbose=0, 

#                    callbacks=[model_checkpoint_callback], 

#                    shuffle = True)

#save_history(history, str('longrun-' + filename))



t1 = time.time()

total = t1-t0

print(str(EPOCHS), 'epochs of training finished in', 

      '{0:.2f}'.format(total), 'seconds')
filename = filename.replace(',', '') # Needed on Kaggle

history_234_lr = load_history_kaggle(str('longrun-' + filename))

plot_history(history_234_lr, ylim = (0,0.1))
# Client path (uncomment to use):

#path_to_best_model = 'models/longruns/sweden_cnn_lstm-066_bsize-8_drop-0.5_conv_sz-(5, 5)_lstm_sz-200_dense_sz10_epoch-593_loss-0.01946.h5'



# Kaggle path (uncomment to use):

path_to_best_model = '../input/covid19-forecasting-models-histories/models/longruns/sweden_cnn_lstm-066_bsize-8_drop-0.5_conv_sz-(5 5)_lstm_sz-200_dense_sz10_epoch-593_loss-0.01946.h5'



model = load_model(path_to_best_model)

model.summary()
test_loss = model.evaluate(X_test, Y_test)



print('{0:.3f}'.format((test_loss-0.01946)/0.01946)) #0.01946 is the validation loss of this model

Y_hat = model.predict(X_test + np.random.normal(0,0.5,X_test.shape), verbose=1)

print(Y_hat.shape)
def plot_prediction(X_test, Y_test, Y_hat, test_idx):



    X_test = unscale(X_test, 'x')

    Y_hat = unscale(Y_hat, 'y')

    Y_test = unscale(Y_test, 'y')

    

    X_test = np.array(X_test[test_idx,:,:], copy = True)

    Y_hat = np.array(Y_hat[test_idx,:,:], copy = True)

    Y_test = np.array(Y_test[test_idx,:,:], copy = True)



    n_obs = len(X_test)

    n_pred = len(Y_hat)

    

    fig, ax = plt.subplots(1, 1, figsize = (15,10))

    

    ax.set_xlim(0, n_obs + n_pred)



    colors = ('blue', 'orange', 'red')

    ax.plot(np.arange(n_obs), X_test[:,0], label = 'ConfirmedCases', color = colors[0])

    ax.plot(np.arange(n_obs), X_test[:,1], label = 'Fatalities', color = colors[1])



    ax.axvline(x = n_obs - 1, ls = '--', color = 'grey')



    ax.plot(np.arange(n_obs - 1, n_obs + n_pred), 

                np.concatenate((np.expand_dims(X_test[-1,0], 0), Y_test[:,0])), 

                color = colors[0])

    ax.plot(np.arange(n_obs - 1, n_obs + n_pred), 

                np.concatenate((np.expand_dims(X_test[-1,1], 0), Y_test[:,1])), 

                color = colors[1])



    ax.plot(np.arange(n_obs - 1, n_obs + n_pred), 

                np.concatenate((np.expand_dims(X_test[-1,0], 0), Y_hat[:,0])), 

                color = colors[2], ls = '--', label = 'Predicted ConfirmedCases')

    ax.plot(np.arange(n_obs - 1, n_obs + n_pred), 

                np.concatenate((np.expand_dims(X_test[-1,1], 0), Y_hat[:,1])), 

                color = colors[2], ls = '--', label = 'Predicted Fatalities')

    ax.legend()

    

for i in range(Y_hat.shape[0]):

    plot_prediction(X_test, Y_test, Y_hat, i)
random_perturbs = np.random.normal(0, 0.25, X_test.shape)

Y_hat = model.predict(X_test + random_perturbs, verbose=1)

print(Y_hat.shape)

    

for i in range(Y_hat.shape[0]):

    plot_prediction(X_test + random_perturbs, Y_test, Y_hat, i)