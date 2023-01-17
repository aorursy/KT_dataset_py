import warnings

warnings.filterwarnings('ignore')



#the basics

import pandas as pd, numpy as np, seaborn as sns

import math, json, os, random

from matplotlib import pyplot as plt

from tqdm import tqdm



#tensorflow basics

import tensorflow as tf

import tensorflow_addons as tfa

import keras.backend as K



from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from sklearn.cluster import KMeans
def seed_everything(seed = 34):

    os.environ['PYTHONHASHSEED']=str(seed)

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    

seed_everything()
#get comp data

train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')

aug_df = pd.read_csv('../input/aug-data/aug_data.csv')
#Exploring signal_to_noise and SN_filter distributions

fig, ax = plt.subplots(1, 2, figsize = (15, 5))

sns.kdeplot(train['signal_to_noise'], shade = True, ax = ax[0])

sns.countplot(train['SN_filter'], ax = ax[1])



ax[0].set_title('Signal/Noise Distribution')

ax[1].set_title('Signal/Noise Filter Distribution');
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

    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]

    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]

    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea], 2)
def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return K.sqrt(mse)



def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):

    score = 0

    for i in range(num_scored):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

    return score
# additional features

def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

    return bpps_arr



def read_bpps_nb(df):

    # from https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm 

    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data

    bpps_nb_std = 0.08914   # std of bpps_nb across all training data

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        bpps_arr.append(bpps_nb)

    return bpps_arr 



train['bpps_sum'] = read_bpps_sum(train)

test['bpps_sum'] = read_bpps_sum(test)

train['bpps_max'] = read_bpps_max(train)

test['bpps_max'] = read_bpps_max(test)

train['bpps_nb'] = read_bpps_nb(train)

test['bpps_nb'] = read_bpps_nb(test)
# clustering for  GroupKFold

kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train)[:,:,0])

train['cluster_id'] = kmeans_model.labels_
def gru_layer(hidden_dim, dropout):

    return tf.keras.layers.Bidirectional(

                                tf.keras.layers.GRU(hidden_dim,

                                dropout=dropout,

                                return_sequences=True,

                                kernel_initializer='orthogonal'))



def lstm_layer(hidden_dim, dropout):

    return tf.keras.layers.Bidirectional(

                                tf.keras.layers.LSTM(hidden_dim,

                                dropout=dropout,

                                return_sequences=True,

                                kernel_initializer='orthogonal'))



def build_model(gru=1, seq_len=107, pred_len=68, dropout=0.5,

                embed_dim=100, hidden_dim=128, layers=3):

    

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))



    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    

    hidden = tf.keras.layers.SpatialDropout1D(.2)(reshaped)  

    

    

    if gru==1:

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        

    elif gru==0:

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        

    elif gru==3:

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        

    elif gru==4:

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        

    elif gru==5:

        hidden = lstm_layer(hidden_dim, dropout)(hidden)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

        hidden = gru_layer(hidden_dim, dropout)(hidden)

    

    #only making predictions on the first part of each sequence

    truncated = hidden[:, :pred_len]

    

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)



    #some optimizers

    adam = tf.optimizers.Adam()

    radam = tfa.optimizers.RectifiedAdam()

    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)

    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)

    

    model.compile(optimizer=adam, loss=mcrmse)

    

    return model
def aug_data(df):

    target_df = df.copy()

    new_df = aug_df[aug_df['id'].isin(target_df['id'])]

                         

    del target_df['structure']

    del target_df['predicted_loop_type']

    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')



    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])

    df['log_gamma'] = 100

    df['score'] = 1.0

    df = df.append(new_df[df.columns])

    return df

train = aug_data(train)

test = aug_data(test)
#basic training configuration

FOLDS = 5

EPOCHS = 75

REPEATS = 1

BATCH_SIZE = 64

VERBOSE = 2

SEED = 34
#get different test sets and process each

public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
# gru_histories = []

# gru_private_preds = np.zeros((private_df.shape[0], 130, 5))

# gru_public_preds = np.zeros((public_df.shape[0], 107, 5))



# gkf = GroupKFold(n_splits=FOLDS)

# for cv, (tr_idx, vl_idx) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):



#     sv_gru = tf.keras.callbacks.ModelCheckpoint(f'gru-{cv}.h5')



#     trn = train.iloc[tr_idx]

#     x_trn = preprocess_inputs(trn)

#     y_trn = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))

#     w_trn = np.log(trn.signal_to_noise+1.1)/2



#     val = train.iloc[vl_idx]

#     x_val_all = preprocess_inputs(val)

#     val = val[val.SN_filter == 1]

#     x_val = preprocess_inputs(val)

#     y_val = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))

#     gru = build_model(gru=1)

#     history = gru.fit(x_trn, y_trn, 

#                       validation_data=(x_val, y_val),

#                       batch_size=BATCH_SIZE,

#                       sample_weight=w_trn/2,

#                       epochs=EPOCHS,

#                       callbacks=[lr_callback,sv_gru],

#                       verbose = VERBOSE)  



#     gru_histories.append(history)



#     #load best model and predict

#     gru_short = build_model(gru=1, seq_len=107, pred_len=107)

#     gru_short.load_weights(f'gru-{cv}.h5')

#     gru_public_pred = gru_short.predict(public_inputs) / FOLDS * REPEATS



#     gru_long = build_model(gru=1, seq_len=130, pred_len=130)

#     gru_long.load_weights(f'gru-{cv}.h5')

#     gru_private_pred = gru_long.predict(private_inputs) / FOLDS * REPEATS



#     gru_public_preds += gru_public_pred

#     gru_private_preds += gru_private_pred



#     del gru_short, gru_long
lstm_histories = []

lstm_private_preds = np.zeros((private_df.shape[0], 130, 5))

lstm_public_preds = np.zeros((public_df.shape[0], 107, 5))



gkf = GroupKFold(n_splits=FOLDS)

for cv, (tr_idx, vl_idx) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):



    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'lstm-{cv}.h5')



    trn = train.iloc[tr_idx]

    x_trn = preprocess_inputs(trn)

    y_trn = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))

    w_trn = np.log(trn.signal_to_noise+1.1)/2



    val = train.iloc[vl_idx]

    x_val_all = preprocess_inputs(val)

    val = val[val.SN_filter == 1]

    x_val = preprocess_inputs(val)

    y_val = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))

    lstm = build_model(gru=1)

    history = lstm.fit(x_trn, y_trn, 

                      validation_data=(x_val, y_val),

                      batch_size=BATCH_SIZE,

                      sample_weight=w_trn/2,

                      epochs=EPOCHS,

                      callbacks=[lr_callback,sv_gru],

                      verbose = VERBOSE)  



    lstm_histories.append(history)



    #load best model and predict

    lstm_short = build_model(gru=1, seq_len=107, pred_len=107)

    lstm_short.load_weights(f'lstm-{cv}.h5')

    lstm_public_pred = lstm_short.predict(public_inputs) / FOLDS * REPEATS



    lstm_long = build_model(gru=1, seq_len=130, pred_len=130)

    lstm_long.load_weights(f'lstm-{cv}.h5')

    lstm_private_pred = lstm_long.predict(private_inputs) / FOLDS * REPEATS



    lstm_public_preds += lstm_public_pred

    lstm_private_preds += lstm_private_pred



    del lstm_short, lstm_long
fig, ax = plt.subplots(1, 2, figsize = (20, 10))



# for history in gru_histories:

#     ax[0].plot(history.history['loss'], color='C0')

#     ax[0].plot(history.history['val_loss'], color='C1')

for history in lstm_histories:

    ax[1].plot(history.history['loss'], color='C0')

    ax[1].plot(history.history['val_loss'], color='C1')



# ax[0].set_title('GRU')

ax[1].set_title('LSTM')



# ax[0].legend(['train', 'validation'], loc = 'upper right')

ax[1].legend(['train', 'validation'], loc = 'upper right')



# ax[0].set_ylabel('MCRMSE')

# ax[0].set_xlabel('Epoch')

ax[1].set_ylabel('MCRMSE')

ax[1].set_xlabel('Epoch');
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
# preds_gru = []



# for df, preds in [(public_df, gru_public_preds), (private_df, gru_private_preds)]:

#     for i, uid in enumerate(df.id):

#         single_pred = preds[i]



#         single_df = pd.DataFrame(single_pred, columns=target_cols)

#         single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



#         preds_gru.append(single_df)



# preds_gru_df = pd.concat(preds_gru).groupby('id_seqpos').mean().reset_index()

# preds_gru_df.head()
preds_lstm = []



for df, preds in [(public_df, lstm_public_preds), (private_df, lstm_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_lstm.append(single_df)



preds_lstm_df = pd.concat(preds_lstm).groupby('id_seqpos').mean().reset_index()

preds_lstm_df.head()
# blend_preds_df = pd.DataFrame()

# blend_preds_df['id_seqpos'] = preds_gru_df['id_seqpos']

# blend_preds_df['reactivity'] = 0.5*preds_gru_df['reactivity'] + 0.5*preds_lstm_df['reactivity']

# blend_preds_df['deg_Mg_pH10'] = 0.5*preds_gru_df['deg_Mg_pH10'] + 0.5*preds_lstm_df['deg_Mg_pH10']

# blend_preds_df['deg_pH10'] = 0.5*preds_gru_df['deg_pH10'] + 0.5*preds_lstm_df['deg_pH10']

# blend_preds_df['deg_Mg_50C'] = 0.5*preds_gru_df['deg_Mg_50C'] + 0.5*preds_lstm_df['deg_Mg_50C']

# blend_preds_df['deg_50C'] = 0.5*preds_gru_df['deg_50C'] + 0.5*preds_lstm_df['deg_50C']
submission = sample_sub[['id_seqpos']].merge(preds_lstm_df, on=['id_seqpos'])

#sanity check

submission.head()
submission.to_csv('submission.csv', index=False)

print('Submission saved')