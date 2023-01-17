import warnings

warnings.filterwarnings('ignore')



#the basics

import pandas as pd, numpy as np, seaborn as sns

import math, json

from matplotlib import pyplot as plt

from tqdm import tqdm



#tensorflow deep learning basics

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.nn as nn

import tensorflow.keras.backend as kb



#for model evaluation

from sklearn.model_selection import train_test_split, KFold
#get comp data

train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
print(train.columns)
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
fig, ax = plt.subplots(1, 2, figsize = (15, 5))

sns.kdeplot(train['signal_to_noise'], shade = True, ax = ax[0])

sns.countplot(train['SN_filter'], ax = ax[1])



ax[0].set_title('Signal/Noise Distribution')

ax[1].set_title('Signal/Noise Filter Distribution');
print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")

print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")
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
train_inputs = preprocess_inputs(train.loc[train.SN_filter == 1])

train_labels = np.array(train.loc[train.SN_filter == 1][target_cols].values.tolist()).transpose((0, 2, 1))
train_inputs.shape, train_labels.shape
len(token2int)
def RMSE(y_actual, y_pred, eps=1e-6):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return kb.sqrt(mse + eps)



def MCRMSE(y_actual, y_pred, num_scored=len(target_cols)):

    score = 0

    for i in range(num_scored):

        score += RMSE(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

        

    return score
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

                embed_dim=100, hidden_dim=128):

    

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

    

    model.compile(optimizer = adam, loss=MCRMSE)

    

    return model
#basic training configuration

FOLDS = 10

EPOCHS = 100

BATCH_SIZE = 64

VERBOSE = 2
#get different test sets and process each

public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
if tf.config.list_physical_devices('GPU') is not None:

    print('Training on GPU')
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
gru_histories = []

gru_private_preds = np.zeros((private_df.shape[0], 130, 5))

gru_public_preds = np.zeros((public_df.shape[0], 107, 5))



kfold = KFold(FOLDS, shuffle = True, random_state = 34)



for f, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

    

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'gru-{f}.h5')

    

    train_ = train_inputs[train_index]

    train_labs = train_labels[train_index]

    val_ = train_inputs[val_index]

    val_labs = train_labels[val_index]

        

    gru = build_model(gru=True)

    history = gru.fit(

                        train_, train_labs, 

                        validation_data=(val_,val_labs),

                        batch_size=BATCH_SIZE,

                        epochs=EPOCHS,

                        callbacks=[lr_callback,sv_gru],

                        verbose = VERBOSE)  

    

    gru_histories.append(history)

    

    #load best model and predict

    gru_short = build_model(gru=True, seq_len=107, pred_len=107)

    gru_short.load_weights(f'gru-{f}.h5')

    gru_public_pred = gru_short.predict(public_inputs) / FOLDS

    

    gru_long = build_model(gru=True, seq_len=130, pred_len=130)

    gru_long.load_weights(f'gru-{f}.h5')

    gru_private_pred = gru_long.predict(private_inputs) / FOLDS

    

    gru_public_preds += gru_public_pred

    gru_private_preds += gru_private_pred

    

    del gru_short, gru_long
print(f" GRU mean fold validation loss: {np.mean([min(history.history['val_loss']) for history in gru_histories])}")
lstm_histories = []

lstm_private_preds = np.zeros((private_df.shape[0], 130, 5))

lstm_public_preds = np.zeros((public_df.shape[0], 107, 5))



kfold = KFold(FOLDS, shuffle = True, random_state = 34)



for f, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

    

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'lstm-{f}.h5')

    

    train_ = train_inputs[train_index]

    train_labs = train_labels[train_index]

    val_ = train_inputs[val_index]

    val_labs = train_labels[val_index]

        

    lstm = build_model(gru=False)

    history = lstm.fit(

                        train_, train_labs, 

                        validation_data=(val_,val_labs),

                        batch_size=BATCH_SIZE,

                        epochs=EPOCHS,

                        callbacks=[lr_callback,sv_gru],

                        verbose = VERBOSE)  

    

    lstm_histories.append(history)

    

    #load best model and predict

    lstm_short = build_model(gru=False, seq_len=107, pred_len=107)

    lstm_short.load_weights(f'lstm-{f}.h5')

    lstm_public_pred = lstm_short.predict(public_inputs) / FOLDS

    

    lstm_long = build_model(gru=False, seq_len=130, pred_len=130)

    lstm_long.load_weights(f'lstm-{f}.h5')

    lstm_private_pred = lstm_long.predict(private_inputs) / FOLDS

    

    lstm_public_preds += lstm_public_pred

    lstm_private_preds += lstm_private_pred

    

    del lstm_short, lstm_long
print(f" LSTM mean fold validation loss: {np.mean([min(history.history['val_loss']) for history in lstm_histories])}")
fig, ax = plt.subplots(1, 2, figsize = (20, 10))



for history in gru_histories:

    ax[0].plot(history.history['loss'])

    ax[0].plot(history.history['val_loss'])

for history in lstm_histories:

    ax[1].plot(history.history['loss'])

    ax[1].plot(history.history['val_loss'])



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
prediction = []

# select the more accurate model as the predictor

if np.mean([min(history.history['val_loss']) for history in gru_histories]) < np.mean([min(history.history['val_loss']) for history in lstm_histories]):

    for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:

        for i, uid in enumerate(df.id):

            single_pred = preds[i]



            single_df = pd.DataFrame(single_pred, columns=target_cols)

            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



            prediction.append(single_df)

            

    preds_df = pd.concat(prediction)

    preds_df.head()

else:

    for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:

        for i, uid in enumerate(df.id):

            single_pred = preds[i]



            single_df = pd.DataFrame(single_pred, columns=target_cols)

            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



            prediction.append(single_df)



    preds_df = pd.concat(prediction)

    preds_df.head()
# preds_gru = []



# for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:

#     for i, uid in enumerate(df.id):

#         single_pred = preds[i]



#         single_df = pd.DataFrame(single_pred, columns=target_cols)

#         single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



#         preds_gru.append(single_df)



# preds_gru_df = pd.concat(preds_gru)

# preds_gru_df.head()
# preds_lstm = []



# for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:

#     for i, uid in enumerate(df.id):

#         single_pred = preds[i]



#         single_df = pd.DataFrame(single_pred, columns=target_cols)

#         single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



#         preds_lstm.append(single_df)



# preds_lstm_df = pd.concat(preds_lstm)

# preds_lstm_df.head()
# blend_preds_df = pd.DataFrame()

# blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']

# blend_preds_df['reactivity'] = .5*preds_gru_df['reactivity'] + .5*preds_lstm_df['reactivity']

# blend_preds_df['deg_Mg_pH10'] = .5*preds_gru_df['deg_Mg_pH10'] + .5*preds_lstm_df['deg_Mg_pH10']

# blend_preds_df['deg_pH10'] = .5*preds_gru_df['deg_pH10'] + .5*preds_lstm_df['deg_pH10']

# blend_preds_df['deg_Mg_50C'] = .5*preds_gru_df['deg_Mg_50C'] + .5*preds_lstm_df['deg_Mg_50C']

# blend_preds_df['deg_50C'] = .5*preds_gru_df['deg_50C'] + .5*preds_lstm_df['deg_50C']
# submission = sample_sub[['id_seqpos']].merge(blend_preds_df, on=['id_seqpos'])

submission = sample_sub[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

#sanity check

submission.head()
submission.to_csv('submission.csv', index=False)

print('Submission saved')