import pandas as pd

import numpy as np

import json

import tensorflow.keras.layers as L

from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Layer

import keras.backend as K

import tensorflow as tf

import plotly.express as px

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from sklearn.cluster import KMeans

import os

import random



EPOCHS = 60

BATCH_SIZE = 64

SEED = 34

TTA = True

FOLD_N = 5
def seed_everything(seed = SEED):

    os.environ['PYTHONHASHSEED']=str(seed)

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

seed_everything()
def pandas_list_to_array(df):    

    return np.transpose(

        np.array(df.values.tolist()),

        (0, 2, 1)

    )



def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):

    base_feature = np.transpose(df[cols]

                                .applymap(lambda seq: [token2int[x] for x in seq])

                                .values.tolist(), 

                                (0, 2, 1))

    bpps_sum_feature = np.array(df['bpps_sum'].tolist())[:, :, np.newaxis]

    bpps_max_feature = np.array(df['bpps_max'].tolist())[:, :, np.newaxis]

    bpps_nb_feature = np.array(df['bpps_nb'].tolist())[:, :, np.newaxis]



    return np.concatenate([base_feature, bpps_sum_feature, bpps_max_feature, bpps_nb_feature], axis=2)



def augment_data(augment_df, df):

    target_df = df.copy()

    new_df = augment_df[augment_df['id'].isin(target_df['id'])]



    del target_df['structure']

    del target_df['predicted_loop_type']



    new_df = new_df.merge(target_df, on=['id', 'sequence'], how='left')



    df['cnt'] = df['id'].map(new_df[['id', 'cnt']].set_index('id').to_dict()['cnt'])

    df['log_gamma'] = 100

    df['score'] = 1.0

    df = df.append(new_df[df.columns])



    return df
# Loss function

def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return tf.keras.backend.sqrt(mse)

    

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):

    score = 0

    for i in range(num_scored):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

    return score
# Inherit tensorflow.keras.layers.Layer

class GRU_Layer(Layer):

    def __init__(self, hidden_dim, dropout):



        super(GRU_Layer, self).__init__()



        self.custom_layers = Bidirectional(GRU(hidden_dim, 

                                            dropout=dropout, 

                                            return_sequences=True, 

                                            kernel_initializer='orthogonal'))

    def call(self, inputs):

        return self.custom_layers(inputs)





class LSTM_Layer(Layer):

    def __init__(self, hidden_dim, dropout):



        super(LSTM_Layer, self).__init__()

        self.custom_layers = Bidirectional(LSTM(hidden_dim, 

                                            dropout=dropout, 

                                            return_sequences=True, 

                                            kernel_initializer='orthogonal'))



    def call(self, inputs):

        return self.custom_layers(inputs)
# Based on @tito model 

class MIX_LSTM_GRU(tf.keras.Model):

    def __init__(self, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, 

                hidden_dim=256, model_type=0, token2int=None):



        super(MIX_LSTM_GRU, self).__init__()



        self.token2int = token2int

        self.embed = L.Embedding(input_dim=len(self.token2int), output_dim=embed_dim)

        self.dense = L.Dense(5, activation='linear')

        



        self.pred_len = pred_len

        self.hidden_dim = hidden_dim

        self.dropout = dropout





        self.type = model_type



    def build(self, inputs):



        self.gru_layers = []

        self.lstm_layers = []

        for i in range(2):

            self.gru_layers.append(GRU_Layer(self.hidden_dim, self.dropout))

            self.lstm_layers.append(LSTM_Layer(self.hidden_dim, self.dropout))



    def call(self, inputs):

        categorical_features = inputs[:, :, :3]

        numerical_features = inputs[:, :, 3:]



        embed = self.embed(categorical_features)

        reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

        reshaped = tf.keras.layers.concatenate([reshaped, numerical_features], axis=2)



        if self.type == 0:

            hidden = self.gru_layers[0](reshaped)

            hidden = self.gru_layers[1](hidden)

        elif self.type == 1:

            hidden = self.lstm_layers[0](reshaped)

            hidden = self.gru_layers[0](hidden)

        elif self.type == 2:

            hidden = self.gru_layers[1](reshaped)

            hidden = self.lstm_layers[1](hidden)

        elif self.type == 3:

            hidden = self.lstm_layers[0](reshaped)

            hidden = self.lstm_layers[1](hidden)        



        truncated = hidden[:, :self.pred_len]

        out = self.dense(truncated)



        return out
def read_bpps_sum(df, root_path):

    bpps_arr = []

    for obj_id in df.id.to_list():

        bppm = np.load(os.path.join(root_path, obj_id + '.npy')).sum(axis=1)

        bpps_arr.append(bppm)

    return bpps_arr





def read_bpps_max(df, root_path):

    bpps_arr = []

    for obj_id in df.id.to_list():

        bppm = np.load(os.path.join(root_path, obj_id + '.npy'))

        bpps_arr.append(np.max(bppm, axis=1))

    return bpps_arr







def read_bpps_nb(df, root_path, bpps_mean, bpps_std):

    bpps_arr = []

    for obj_id in df.id.to_list():

        bpps = np.load(os.path.join(root_path, obj_id + '.npy'))

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_nb = (bpps_nb - bpps_mean) / bpps_std

        bpps_arr.append(bpps_nb)

    

    return bpps_arr





def calc_bpps_mean_std(df, root_path):

    bpps_arr = []

    for obj_id in df.id.to_list():

        bpps = np.load(os.path.join(root_path, obj_id + '.npy'))

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_arr.append(bpps_nb)

    



    return np.mean(bpps_arr), np.std(bpps_arr)



def add_bpp_feature(train, test, bpps_dir):

    mean, std = calc_bpps_mean_std(train, bpps_dir)

    train['bpps_sum'] = read_bpps_sum(train, bpps_dir)

    test['bpps_sum'] = read_bpps_sum(test, bpps_dir)

    train['bpps_max'] = read_bpps_max(train, bpps_dir)

    test['bpps_max'] = read_bpps_max(test, bpps_dir)

    train['bpps_nb'] = read_bpps_nb(train, bpps_dir, mean, std)

    test['bpps_nb'] = read_bpps_nb(test, bpps_dir, mean, std)



    return train, test
def group_by_kmean_reactivity(train, token2int):

    kmeans_model = KMeans(n_clusters=200, random_state=110).fit(preprocess_inputs(train, token2int)[:, :, 0])

    train['cluster_id'] = kmeans_model.labels_

    

    return train
def train_mix_lstm_gru(train, public_df, private_df, target_cols, token2int, model_type=0, FOLD_N=FOLD_N):

    Ver='MIX_LSTM_GRU_' + str(model_type) 

    gkf = GroupKFold(n_splits=FOLD_N)



    public_inputs = preprocess_inputs(public_df, token2int)

    private_inputs = preprocess_inputs(private_df, token2int)





    holdouts = []

    holdout_preds = []



    for cv, (tr_idx, vl_idx) in enumerate(gkf.split(train,  train['reactivity'], train['cluster_id'])):

        trn = train.iloc[tr_idx]

        x_trn = preprocess_inputs(trn, token2int)

        y_trn = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))

        w_trn = np.log(trn.signal_to_noise+1.1)/2



        val = train.iloc[vl_idx]

        x_val_all = preprocess_inputs(val, token2int)

        val = val[val.SN_filter == 1]

        x_val = preprocess_inputs(val, token2int)

        y_val = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))



        model = MIX_LSTM_GRU(model_type=model_type, token2int=token2int)

        model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)



        model_short = MIX_LSTM_GRU(seq_len=107, pred_len=107, model_type=model_type, token2int=token2int)

        model_long = MIX_LSTM_GRU(seq_len=130, pred_len=130, model_type=model_type, token2int=token2int)



        history = model.fit(

            x_trn, y_trn,

            validation_data = (x_val, y_val),

            batch_size=BATCH_SIZE,

            epochs=EPOCHS,

            sample_weight=w_trn,

            callbacks=[

                tf.keras.callbacks.ReduceLROnPlateau(),

                tf.keras.callbacks.ModelCheckpoint(f'model{Ver}_cv{cv}.h5')

            ]

        )



        fig = px.line(

            history.history, y=['loss', 'val_loss'], 

            labels={'index': 'epoch', 'value': 'Mean Squared Error'}, 

            title='Training History')

        fig.show()



        model.load_weights(f'model{Ver}_cv{cv}.h5')

        

        model_short.compile(optimizer=tf.optimizers.Adam(), loss=mcrmse)

        model_short.train_on_batch(tf.zeros((1,107,6)), tf.zeros((1,107,5)))

        model_short.load_weights(f'model{Ver}_cv{cv}.h5')



        model_long.compile(optimizer=tf.optimizers.Adam(), loss=mcrmse)

        model_long.train_on_batch(tf.zeros((1,130,6)), tf.zeros((1,130,5)))        

        model_long.load_weights(f'model{Ver}_cv{cv}.h5')





        holdouts.append(train.iloc[vl_idx])

        holdout_preds.append(model.predict(x_val_all))

        if cv == 0:

            public_preds = model_short.predict(public_inputs)/FOLD_N

            private_preds = model_long.predict(private_inputs)/FOLD_N

        else:

            public_preds += model_short.predict(public_inputs)/FOLD_N

            private_preds += model_long.predict(private_inputs)/FOLD_N

    return holdouts, holdout_preds, public_df, public_preds, private_df, private_preds
data_dir = '/kaggle/input/stanford-covid-vaccine/'

bpps_dir = '/kaggle/input/stanford-covid-vaccine/bpps/'

train = pd.read_json(data_dir + 'train.json', lines=True)

test = pd.read_json(data_dir + 'test.json', lines=True)

sample_df = pd.read_csv(data_dir + 'sample_submission.csv')

aug_df = pd.read_csv('/kaggle/input/openvaccine-augment/aug_data.csv')



token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train, test = add_bpp_feature(train, test, bpps_dir)

train = group_by_kmean_reactivity(train, token2int)



if TTA:

    train = augment_data(aug_df, train)

    test = augment_data(aug_df, test)





target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

train_inputs = preprocess_inputs(train, token2int)

train_labels = pandas_list_to_array(train[target_cols])

public_df = test.query("seq_length == 107")

private_df = test.query("seq_length == 130")
val_df, val_preds, test_df, test_preds = [], [], [], []

debug = False

if debug:

    nmodel = 1

else:

    nmodel = 4

for i in range(nmodel):

    holdouts, holdout_preds, public_df, public_preds, private_df, private_preds = train_mix_lstm_gru(train, 

                                                                                                     public_df, private_df, 

                                                                                                     target_cols, 

                                                                                                     token2int,model_type=i)

    val_df += holdouts

    val_preds += holdout_preds

    test_df.append(public_df)

    test_df.append(private_df)

    test_preds.append(public_preds)

    test_preds.append(private_preds)
preds_ls = []

for df, preds in zip(test_df, test_preds):

    for i, uid in enumerate(df.id):

        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()

# .mean() is for

# 1, Predictions from multiple models

# 2, TTA (augmented test data)



preds_ls = []

for df, preds in zip(val_df, val_preds):

    for i, uid in enumerate(df.id):

        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=target_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        single_df['SN_filter'] = df[df['id'] == uid].SN_filter.values[0]

        preds_ls.append(single_df)

holdouts_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()
submission = preds_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

submission.to_csv(f'submission.csv', index=False)

print(f'wrote to submission.csv')