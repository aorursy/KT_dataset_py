# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Get version python/keras/tensorflow/sklearn

from platform import python_version

import sklearn

import keras

import tensorflow as tf



# Folder manipulation

import os



# Garbage collector

import gc



# Linear algebra and data processing

import numpy as np

import pandas as pd

from pandas import datetime



# Visualisation of picture and graph

import matplotlib.pyplot as plt

import seaborn as sns



# Keras importation

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import Input, Dense, RNN, Bidirectional, concatenate, GRUCell, LSTMCell

from keras.models import Model, Sequential

from keras.optimizers import Adam



# Others

from tqdm import tqdm, tqdm_notebook
print(os.listdir("../input"))

print("Keras version : " + keras.__version__)

print("Tensorflow version : " + tf.__version__)

print("Python version : " + python_version())

print("Sklearn version : " + sklearn.__version__)
MAIN_DIR = "../input/bitcoin-historical-data/"

DATA = "bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv"



# Size of the dataset in percentage

TEST_SIZE = 5

VAL_SIZE = 5



OUTPUT_SIZE = 30

INPUT_SIZE = 30



# Set graph font size

sns.set(font_scale=1.3)
def load_data():

    df = pd.read_csv(MAIN_DIR+DATA)

    # We don't take all the data in order to reduce training time

    df = df[-25000:].reset_index(drop=True)

    return df
data_raw = load_data()
print(f"Shape of dataset : {data_raw.shape}")
data_raw.head()
data_raw.isna().sum()
data_raw = data_raw.dropna()
def plot_curves(df, var_y, var_x='MJD'):

    fig, ax = plt.subplots(figsize=(16,5))

    sns.lineplot(x=var_x,y=var_y, data=df, ax=ax)



    ax.set_title(f"'{var_y}' value evolution in function of time")

    ax.set_xlabel(f'{var_x}')

    ax.set_ylabel(f"'{var_y}' value")
plot_curves(data_raw, 'Weighted_Price', 'Timestamp')
def feature_engineering(data):

    drop_feat = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(BTC)',

       'Volume_(Currency)']

    

    df = data.copy()

    

    # Drop useless feature

    df = df.drop(drop_feat, axis=1)

    df.columns = ['target']

    

    # Drop Nan for simplicity

    df = df.dropna()

    

    return df
data_raw = load_data()

data = feature_engineering(data_raw)
data.head()
print(f"New dataset shape : {data.shape}")
# Get index limit of train/val/test index in the dataset

def get_limit_split(data, val_size, test_size, output_size):

    # Convert percentage into value

    val_size = int((val_size*0.01)*data.shape[0])

    test_size = int((test_size*0.01)*data.shape[0])



    limit_train = data.shape[0] - val_size - test_size - output_size + 1

    limit_val = limit_train + val_size

    limit_test = limit_val + test_size

    

    return limit_train, limit_val, limit_test
# It would be better to use Keras or Sklearn normalize implementation...

def normalize(data, val_size, test_size, output_size):

    def apply(X, mean, std):

        X = (X - mean) / std

        return X

    

    df = data.copy()

    

    val_size = int((val_size*0.01)*df.shape[0])

    test_size = int((test_size*0.01)*df.shape[0])

    

    limit_train = df.shape[0] - val_size - test_size - output_size + 1

    limit_val = limit_train + val_size

    limit_test = limit_val + test_size

    

    mean = df.iloc[0:limit_train]['target'].mean()

    std = df.iloc[0:limit_train]['target'].std()

    

    df.iloc[0:limit_train]['target'] = apply(df.iloc[0:limit_train]['target'].values, mean, std)

    df.iloc[limit_train:limit_val]['target'] = apply(df.iloc[limit_train:limit_val]['target'].values, mean, std)

    df.iloc[limit_val:limit_test]['target'] = apply(df.iloc[limit_val:limit_test]['target'].values, mean, std)

    

    return df, mean, std
# It would be better to use Keras or Sklearn normalize implementation...

def denormalize(X, mean, std):

    X = (X * std) + mean

    return X
def plot_data_split(data, val_size, test_size, output_size):

    df = data.copy()

    

    limit_train, limit_val, limit_test = get_limit_split(df, val_size, test_size, output_size)

    

    df.at[0:limit_train, 'dataset'] = 'train'

    df.at[limit_train:limit_val, 'dataset'] = 'val'

    df.at[limit_val:limit_test, 'dataset'] = 'test'

    

    fig, ax = plt.subplots(figsize=(16,5))

    sns.lineplot(x=df.index,y='target', data=df, ax=ax, hue='dataset')



    ax.set_title(f"Bitcoin value evolution in function of time")

    ax.set_xlabel(f'Index in dataset')

    ax.set_ylabel(f"Bicoin value")
def train_val_test_split(X, y, val_size=10, test_size=10, input_size=1, output_size=0):

    # Convert percentage into value

    val_size = int((val_size*0.01)*X.shape[0])

    test_size = int((test_size*0.01)*X.shape[0])

    

    limit_train = X.shape[0] - val_size - test_size - output_size + 1

    limit_val = limit_train + val_size

    limit_test = limit_val + test_size

    

    # TRAINING SET

    X_train = []

    y_train = []

    for i in range(input_size,limit_train):

        X_train.append(X[i-input_size:i,:])

        y_train.append(y[i:i+output_size,:])

    X_train, y_train = np.array(X_train), np.array(y_train)

    

    # VALIDATION SET

    X_val = []

    y_val = []

    for i in range(limit_train,limit_val):

        X_val.append(X[i-input_size:i,:])

        y_val.append(y[i:i+output_size,:])

    X_val, y_val = np.array(X_val), np.array(y_val)

    

    # TEST SET

    X_test = []

    y_test = []

    for i in range(limit_val,limit_test):

        X_test.append(X[i-input_size:i,:])

        y_test.append(y[i:i+output_size,:])

    X_test, y_test = np.array(X_test), np.array(y_test)

    

    return X_train, y_train, X_val, y_val, X_test, y_test
# Normalize data

data_norm, mean, std = normalize(data, val_size=VAL_SIZE, test_size=TEST_SIZE, output_size=OUTPUT_SIZE)
# Reshape data and get different set (train, validation and test set)

X_norm = data_norm.values

X_train, y_train, X_val, y_val, _, _= train_val_test_split(X_norm, X_norm, 

                                                           val_size=VAL_SIZE, 

                                                           test_size=TEST_SIZE, 

                                                           input_size=INPUT_SIZE, 

                                                           output_size=OUTPUT_SIZE)
print(f"X_train shape : {X_train.shape}")

print(f"y_train shape : {y_train.shape}")
plot_data_split(data, val_size=VAL_SIZE, test_size=TEST_SIZE, output_size=OUTPUT_SIZE)
def build_model(layers, n_in_features=1, n_out_features=1, gru=False, bidirectional=False):

    

    keras.backend.clear_session()

    

    n_layers = len(layers)

    

    ######################

    # MODEL

    ######################

    

    ## Encoder

    encoder_inputs = Input(shape=(None, n_in_features))

    

    if(gru):

        rnn_cells = [GRUCell(hidden_dim) for hidden_dim in layers]

    else:

        rnn_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]

        

    if bidirectional:

        encoder = Bidirectional(RNN(rnn_cells, return_state=True), merge_mode=None)

        

        encoder_outputs_and_states = encoder(encoder_inputs)

        encoder_states = []

        

        if(gru):

            bi_encoder_states = encoder_outputs_and_states[2:]

            sep_states = int(len(bi_encoder_states)/2)

        

            for i in range(0, sep_states):

                temp = concatenate([bi_encoder_states[i],bi_encoder_states[sep_states + i]], axis=-1)

                encoder_states.append(temp)

        else:

            bi_encoder_states = encoder_outputs_and_states[2:]

            sep_states = int(len(bi_encoder_states)/2)

            

            for i in range(sep_states):

                temp = concatenate([bi_encoder_states[i],bi_encoder_states[2*n_layers + i]], axis=-1)

                encoder_states.append(temp)

        

    else:  

        encoder = RNN(rnn_cells, return_state=True)

        encoder_outputs_and_states = encoder(encoder_inputs)

        encoder_states = encoder_outputs_and_states[1:]

    

    ## Decoder

    decoder_inputs = Input(shape=(None, n_out_features))

    

    if(gru):

        if bidirectional:

            decoder_cells = [GRUCell(hidden_dim*2) for hidden_dim in layers]

        else:

            decoder_cells = [GRUCell(hidden_dim) for hidden_dim in layers]

    else:

        if bidirectional:

            decoder_cells = [LSTMCell(hidden_dim*2) for hidden_dim in layers]

        else:

            decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]

        

    decoder = RNN(decoder_cells, return_sequences=True, return_state=True)



    decoder_outputs_and_states = decoder(decoder_inputs,

                                         initial_state=encoder_states)

    decoder_outputs = decoder_outputs_and_states[0]



    decoder_dense = Dense(n_out_features, activation='linear') 

    decoder_outputs = decoder_dense(decoder_outputs)

    

    model = Model([encoder_inputs,decoder_inputs], decoder_outputs)

    

    ######################

    # INFERENCE ENCODER

    ######################

    

    encoder_model = Model(encoder_inputs, encoder_states)

    

    ######################

    # INFERENCE DECODER

    ######################

    

    if(gru):

        if bidirectional:

            decoder_states_inputs = [Input(shape=(None, hidden_dim*2)) for hidden_dim in layers]

            decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

        else:

            decoder_states_inputs = [Input(shape=(None, hidden_dim)) for hidden_dim in layers]

            decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

    else:

        layers_repeat = np.repeat(np.array(layers), 2)

        if bidirectional:

            decoder_states_inputs = [Input(shape=(None, hidden_dim*2)) for hidden_dim in layers_repeat]

            decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

        else:

            decoder_states_inputs = [Input(shape=(None, hidden_dim)) for hidden_dim in layers_repeat]

            decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

        

    decoder_states = decoder_outputs_and_states[1:]

    decoder_outputs = decoder_outputs_and_states[0]

    

    decoder_outputs = decoder_dense(decoder_outputs)

    

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    

    model.summary()

    

    return model, encoder_model, decoder_model
def predict(infenc, infdec, source, n_steps):

    output = np.empty((0, n_steps, 1), np.float64)

    

    for row in tqdm(range(source.shape[0])):

        

        states = infenc.predict(source[row:row+1])

        states = [np.reshape(state, (1, 1, state.shape[-1])) for state in states]

        

        output_row = np.empty((1, 0, 1), np.float64)

        target_seq = np.zeros((1, 1, 1))

        input_states = [target_seq] + states

    

        for t in range(n_steps):

            output_states = infdec.predict(input_states)

            output_row = np.concatenate((output_row, output_states[0]), axis=1)

        

            # update state

            states = output_states[1:]

            

            # update target sequence

            target_seq = output_states[0]

            input_states = output_states

        

        output = np.concatenate((output, output_row), axis=0)

            

    return output
def train(X_train, y_train, X_val, y_val):

    bidirectional = True

    layers = [512, 512, 512]

    epochs = 20



    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-6, verbose=0),

           EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=10, verbose=1, mode='min', restore_best_weights=True)]#,

    

    model, encoder, decoder = build_model(layers, X_train.shape[2], y_train.shape[2], gru=True, bidirectional=True)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mean_absolute_error', metrics=['mae'])

    

    X_train_bis = np.pad(y_train, ((0, 0), (1, 0), (0, 0)),

                         mode='constant')[:, :-1]

    X_val_bis = np.pad(y_val, ((0, 0), (1, 0), (0, 0)),

                       mode='constant')[:, :-1]

    

    history = model.fit([X_train, X_train_bis], y_train,

                        validation_data=([X_val, X_val_bis],y_val),

                        epochs=epochs,

                        batch_size=32,

                        shuffle=True,

                        callbacks=cbs)

    

    return history, model, encoder, decoder
history, model, encoder, decoder = train(X_train, y_train, X_val, y_val)
# Plotting learning curve

def plot_loss(history):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    

    # Plot train/val MAE

    ax[0].plot(history.history['mean_absolute_error'])

    ax[0].plot(history.history['val_mean_absolute_error'])

    ax[0].set_title('Model accuracy')

    ax[0].set_ylabel('MSE')

    ax[0].set_xlabel('Epochs')

    ax[0].legend(['Train', 'Test'], loc='upper left')

    

    # Plot train/val loss

    ax[1].plot(history.history['loss'])

    ax[1].plot(history.history['val_loss'])

    ax[1].set_title('Model Loss')

    ax[1].set_ylabel('Loss')

    ax[1].set_xlabel('Epochs')

    ax[1].legend(['Train', 'Test'], loc='upper left')
plot_loss(history)
def plot_lr(history, info):

    fig, ax = plt.subplots(figsize=(7, 5))

    

    # Plot learning rate

    ax.plot(history.history['lr'])

    ax.set_title(f"{info} learning rate evolution in function of epoch")

    ax.set_ylabel('Learning rate value')

    ax.set_xlabel('Epochs')

    ax.legend(['Train'], loc='upper right')
plot_lr(history, info="Model")
# Some functions to help out with

def plot_predictions(y_true, y_pred, title, inter_start, inter_end):

    

    if(inter_start and inter_end):

        y_true = y_true.ravel()[inter_start:inter_end]

        y_pred = y_pred.ravel()[inter_start:inter_end]

    

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    

    y_true = y_true.ravel()

    y_pred = y_pred.ravel()

    

    err = np.mean(np.abs(y_true - y_pred))

    

    ax.plot(y_true)

    ax.plot(y_pred)

    ax.set_title(f"{title} : {err} (MAE)")

    ax.set_ylabel('Value')

    ax.set_xlabel('Index')

    ax.legend(['Real', 'Predict'], loc='upper left')
def reshape(X):

    return np.reshape(X,(1, X.shape[0], 1))
limit_train, limit_val, limit_test = get_limit_split(data, 

                                                     val_size=VAL_SIZE, 

                                                     test_size=TEST_SIZE, 

                                                     output_size=OUTPUT_SIZE)
input_val_norm = reshape(X_norm[limit_train-OUTPUT_SIZE:limit_train])

input_test_norm = reshape(X_norm[limit_val-OUTPUT_SIZE:limit_val])



# Make prediction

y_pred_val_normalize = predict(infenc=encoder, 

                               infdec=decoder, 

                               source=input_val_norm, 

                               n_steps=limit_val-limit_train)

y_pred_test_normalize = predict(infenc=encoder, 

                                infdec=decoder, 

                                source=input_test_norm, 

                                n_steps=limit_test-limit_val)



# Denormalize data

y_pred_val = denormalize(y_pred_val_normalize, mean, std)

y_pred_test = denormalize(y_pred_test_normalize, mean, std)

X = denormalize(X_norm, mean, std)
# We perform prediction on all the validation set and compare on all the validation set

plot_predictions(X[limit_train:limit_val], y_pred_val[:,:,0], 'Model', None, None)
# We perform prediction on all the validation set and compare only on the 30 first examples

plot_predictions(X[limit_train:limit_train+30], y_pred_val[:,0:30,0], 'Model', None, None)
# We perform prediction on all the test set and compare on all the test set

plot_predictions(X[limit_val:limit_test], y_pred_test[:,:,0], 'Model', None, None)
# We perform prediction on all the test set and compare only on the 30 first examples

plot_predictions(X[limit_val:limit_val+30], y_pred_test[:,0:30,0], 'Model', None, None)