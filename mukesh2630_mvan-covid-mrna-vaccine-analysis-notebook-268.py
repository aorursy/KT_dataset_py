import warnings
warnings.filterwarnings('ignore')

#Basic data manipulation libraries
import pandas as pd, numpy as np
import math, json, gc, random, os, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

#Deep Learning Libraries
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

#Library for model evaluation
from sklearn.model_selection import train_test_split, KFold
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()),
        (0, 2, 1))

train_inputs = preprocess_inputs(train[train.signal_to_noise > 1])
train_labels = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))
train.head()
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

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

def build_model(gru=1,seq_len=107, pred_len=68, dropout=0.5,
                embed_dim=75, hidden_dim=128):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)
    
    if gru==1:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        
    elif gru==0:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==3:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        
    elif gru==4:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif gru==5:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
    
    #only making predictions on the first part of each sequence
    truncated = hidden[:, :pred_len]
    
    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    #some optimizers
    adam = tf.optimizers.Adam()
    radam = tfa.optimizers.RectifiedAdam()
    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)
    
    model.compile(optimizer = adam, loss=MCRMSE)
    
    return model
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=.1, random_state=34)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()

gru = build_model(gru=1)
sv_gru = tf.keras.callbacks.ModelCheckpoint('model_gru.h5')

history_gru = gru.fit(
    train_inputs, train_labels, 
    validation_data=(val_inputs,val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback,sv_gru],
    verbose = 2
)

print(f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}")
lstm = build_model(gru=0)
sv_lstm = tf.keras.callbacks.ModelCheckpoint('model_lstm.h5')

history_lstm = lstm.fit(
    train_inputs, train_labels, 
    validation_data=(val_inputs,val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback,sv_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}")
lstm = build_model(gru=3)
sv_lstm = tf.keras.callbacks.ModelCheckpoint('model_hyb1.h5')

history_lstm = lstm.fit(
    train_inputs, train_labels, 
    validation_data=(val_inputs,val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback,sv_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}")
lstm = build_model(gru=4)
sv_lstm = tf.keras.callbacks.ModelCheckpoint('model_hyb2.h5')

history_lstm = lstm.fit(
    train_inputs, train_labels, 
    validation_data=(val_inputs,val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback,sv_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}")
lstm = build_model(gru=5)
sv_lstm = tf.keras.callbacks.ModelCheckpoint('model_hyb3.h5')

history_lstm = lstm.fit(
    train_inputs, train_labels, 
    validation_data=(val_inputs,val_labels),
    batch_size=64,
    epochs=100,
    callbacks=[lr_callback,sv_lstm],
    verbose = 2
)

print(f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}")
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

# build all models
gru_short = build_model(gru=1, seq_len=107, pred_len=107)
gru_long = build_model(gru=1, seq_len=130, pred_len=130)
lstm_short = build_model(gru=0, seq_len=107, pred_len=107)
lstm_long = build_model(gru=0, seq_len=130, pred_len=130)
hyb1_short = build_model(gru=3, seq_len=107, pred_len=107)
hyb1_long = build_model(gru=3, seq_len=130, pred_len=130)
hyb2_short = build_model(gru=4, seq_len=107, pred_len=107)
hyb2_long = build_model(gru=4, seq_len=130, pred_len=130)
hyb3_short = build_model(gru=5, seq_len=107, pred_len=107)
hyb3_long = build_model(gru=5, seq_len=130, pred_len=130)


# load pre-trained model weights
gru_short.load_weights('model_gru.h5')
gru_long.load_weights('model_gru.h5')
lstm_short.load_weights('model_lstm.h5')
lstm_long.load_weights('model_lstm.h5')
hyb1_short.load_weights('model_hyb1.h5')
hyb1_long.load_weights('model_hyb1.h5')
hyb2_short.load_weights('model_hyb2.h5')
hyb2_long.load_weights('model_hyb2.h5')
hyb3_short.load_weights('model_hyb3.h5')
hyb3_long.load_weights('model_hyb3.h5')

# and predict
gru_public_preds = gru_short.predict(public_inputs)
gru_private_preds = gru_long.predict(private_inputs)
lstm_public_preds = lstm_short.predict(public_inputs)
lstm_private_preds = lstm_long.predict(private_inputs)
hyb1_public_preds = hyb1_short.predict(public_inputs)
hyb1_private_preds = hyb1_long.predict(private_inputs)
hyb2_public_preds = hyb2_short.predict(public_inputs)
hyb2_private_preds = hyb2_long.predict(private_inputs)
hyb3_public_preds = hyb3_short.predict(public_inputs)
hyb3_private_preds = hyb3_long.predict(private_inputs)

preds_gru = []

for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru)

preds_lstm = []

for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm)


preds_hyb1 = []

for df, preds in [(public_df, hyb1_public_preds), (private_df, hyb1_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_hyb1.append(single_df)

preds_hyb1_df = pd.concat(preds_hyb1)


preds_hyb2 = []

for df, preds in [(public_df, hyb2_public_preds), (private_df, hyb2_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_hyb2.append(single_df)

preds_hyb2_df = pd.concat(preds_hyb2)

preds_hyb3 = []

for df, preds in [(public_df, hyb3_public_preds), (private_df, hyb3_private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_hyb3.append(single_df)

preds_hyb3_df = pd.concat(preds_hyb3)

blend_preds_df = pd.DataFrame()
blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']
blend_preds_df['reactivity'] = 0.2*preds_gru_df['reactivity'] + 0.2*preds_lstm_df['reactivity'] + 0.2*preds_hyb1_df['reactivity'] + 0.2*preds_hyb2_df['reactivity'] + 0.2*preds_hyb3_df['reactivity']
blend_preds_df['deg_Mg_pH10'] = 0.2*preds_gru_df['deg_Mg_pH10'] + 0.2*preds_lstm_df['deg_Mg_pH10'] + 0.2*preds_hyb1_df['deg_Mg_pH10'] + 0.2*preds_hyb2_df['deg_Mg_pH10'] + 0.2*preds_hyb3_df['deg_Mg_pH10']
blend_preds_df['deg_pH10'] = 0.2*preds_gru_df['deg_pH10'] + 0.2*preds_lstm_df['deg_pH10'] + 0.2*preds_hyb1_df['deg_pH10'] + 0.2*preds_hyb2_df['deg_pH10'] + 0.2*preds_hyb3_df['deg_pH10']
blend_preds_df['deg_Mg_50C'] = 0.2*preds_gru_df['deg_Mg_50C'] + 0.2*preds_lstm_df['deg_Mg_50C'] + 0.2*preds_hyb1_df['deg_Mg_50C'] + 0.2*preds_hyb2_df['deg_Mg_50C'] + 0.2*preds_hyb3_df['deg_Mg_50C']
blend_preds_df['deg_50C'] = 0.2*preds_gru_df['deg_50C'] + 0.2*preds_lstm_df['deg_50C'] + 0.2*preds_hyb1_df['deg_50C'] + 0.2*preds_hyb2_df['deg_Mg_50C'] + 0.2*preds_hyb3_df['deg_Mg_50C']

submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])
submission.head()

#Saving the final output file
submission.to_csv('submission.csv', index=False)