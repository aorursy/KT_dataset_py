import warnings

warnings.filterwarnings('ignore')



#the basics

import pandas as pd, numpy as np

import math, json, gc, random, os, sys

from matplotlib import pyplot as plt

from tqdm import tqdm



#tensorflow deep learning basics

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L



#for model evaluation

from sklearn.model_selection import train_test_split, KFold
#get comp data

train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
#sneak peak

print(train.shape)

if ~ train.isnull().values.any(): print('No missing values')

train.head()
#sneak peak

print(test.shape)

if ~ test.isnull().values.any(): print('No missing values')

test.head()
#sneak peak

print(sample_sub.shape)

if ~ sample_sub.isnull().values.any(): print('No missing values')

sample_sub.head()
#target columns

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    return np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )
train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])

train_labels = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))
def gru_layer(hidden_dim, dropout):

    return tf.keras.layers.Bidirectional(

                                tf.keras.layers.GRU(hidden_dim,

                                dropout=dropout,

                                return_sequences=True,

                                kernel_initializer = 'orthogonal'))



def lstm_layer(hidden_dim, dropout):

    return tf.keras.layers.Bidirectional(

                                tf.keras.layers.LSTM(hidden_dim,

                                dropout=dropout,

                                return_sequences=True,

                                kernel_initializer = 'orthogonal'))



def build_model(gru=False,seq_len=107, pred_len=68, dropout=0.5,

                embed_dim=75, hidden_dim=128):

    

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))



    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    

    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)

    

    if gru:

        hidden = gru_layer(hidden_dim, dropout)(reshaped)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        

    else:

        hidden = lstm_layer(hidden_dim, dropout)(reshaped)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    

    #only making predictions on the first part of each sequence

    truncated = hidden[:, :pred_len]

    

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)



    #some optimizers

    adam = tf.optimizers.Adam()

    radam = tfa.optimizers.RectifiedAdam()

    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)

    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)

    

    model.compile(optimizer = adam, loss='mse')

    

    return model
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels,

                                                                     test_size=.1, random_state=34)
if tf.config.list_physical_devices('GPU') is not None:

    print('Training on GPU')
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
gru = build_model(gru=True)

sv_gru = tf.keras.callbacks.ModelCheckpoint('model_gru.h5')



history_gru = gru.fit(

    train_inputs, train_labels, 

    validation_data=(val_inputs,val_labels),

    batch_size=64,

    epochs=70,

    callbacks=[lr_callback,sv_gru],

    verbose = 2

)



print(f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}")
lstm = build_model(gru=False)

sv_lstm = tf.keras.callbacks.ModelCheckpoint('model_lstm.h5')



history_lstm = lstm.fit(

    train_inputs, train_labels, 

    validation_data=(val_inputs,val_labels),

    batch_size=64,

    epochs=75,

    callbacks=[lr_callback,sv_lstm],

    verbose = 2

)



print(f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}")
fig, ax = plt.subplots(1, 2, figsize = (20, 10))



ax[0].plot(history_gru.history['loss'])

ax[0].plot(history_gru.history['val_loss'])



ax[1].plot(history_lstm.history['loss'])

ax[1].plot(history_lstm.history['val_loss'])



ax[0].set_title('GRU')

ax[1].set_title('LSTM')



ax[0].legend(['train', 'validation'], loc = 'upper right')

ax[1].legend(['train', 'validation'], loc = 'upper right')



ax[0].set_ylabel('Loss')

ax[0].set_xlabel('Epoch')

ax[1].set_ylabel('Loss')

ax[1].set_xlabel('Epoch');
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
#build all models

gru_short = build_model(gru=True, seq_len=107, pred_len=107)

gru_long = build_model(gru=True, seq_len=130, pred_len=130)

lstm_short = build_model(gru=False, seq_len=107, pred_len=107)

lstm_long = build_model(gru=False, seq_len=130, pred_len=130)



#load pre-trained model weights

gru_short.load_weights('model_gru.h5')

gru_long.load_weights('model_gru.h5')

lstm_short.load_weights('model_lstm.h5')

lstm_long.load_weights('model_lstm.h5')



#and predict

gru_public_preds = gru_short.predict(public_inputs)

gru_private_preds = gru_long.predict(private_inputs)

lstm_public_preds = lstm_short.predict(public_inputs)

lstm_private_preds = lstm_long.predict(private_inputs)
preds_gru = []



for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_gru.append(single_df)



preds_gru_df = pd.concat(preds_gru)

preds_gru_df.head()
preds_lstm = []



for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_lstm.append(single_df)



preds_lstm_df = pd.concat(preds_lstm)

preds_lstm_df.head()
blend_preds_df = pd.DataFrame()

blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']

blend_preds_df['reactivity'] = .5*preds_gru_df['reactivity'] + .5*preds_lstm_df['reactivity']

blend_preds_df['deg_Mg_pH10'] = .5*preds_gru_df['deg_Mg_pH10'] + .5*preds_lstm_df['deg_Mg_pH10']

blend_preds_df['deg_pH10'] = .5*preds_gru_df['deg_pH10'] + .5*preds_lstm_df['deg_pH10']

blend_preds_df['deg_Mg_50C'] = .5*preds_gru_df['deg_Mg_50C'] + .5*preds_lstm_df['deg_Mg_50C']

blend_preds_df['deg_50C'] = .5*preds_gru_df['deg_50C'] + .5*preds_lstm_df['deg_50C']
submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])



#sanity check

submission.head()
submission.to_csv('submission.csv', index=False)

print('Submission saved')