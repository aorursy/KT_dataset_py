import pandas as pd

import numpy as np

import json

import os

from tqdm import tqdm



# sklearn

from sklearn.model_selection import train_test_split



# tensorflow and keras

from keras.utils.vis_utils import plot_model



import tensorflow.keras.layers as L

import keras.backend as K

import tensorflow as tf
!pip install spektral -q
from spektral.layers import GraphConv
train_json_path = "/kaggle/input/stanford-covid-vaccine/train.json"

test_json_path = "/kaggle/input/stanford-covid-vaccine/test.json"

sample_sub_path = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"



output_path = "./"

bpps_path = "/kaggle/input/stanford-covid-vaccine/bpps"



train_df = pd.read_json(train_json_path, lines=True)

test_df = pd.read_json(test_json_path, lines=True)



# there are 2 part of the test set, they have different seq length

public_df = test_df.query("seq_length == 107").copy()

private_df = test_df.query("seq_length == 130").copy()
train_df.shape
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

train_df[pred_cols].head()
train_y = np.array(train_df[pred_cols].values.tolist()).transpose((0, 2, 1))

train_y.shape
train_df[["id", "sequence", "structure", "predicted_loop_type"]].head()
# label encodings for 

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

sequence_token2int = {x:i for i, x in enumerate('AGUC')}

structure_token2int = {

    '.': 0,

    '(': 1,

    ')': 2,

}

loop_token2int = {x:i for i, x in enumerate('SMIBHEX')}

token2int_map = {

    "sequence": sequence_token2int,

    "structure": structure_token2int,

    "predicted_loop_type": loop_token2int

}

sequence_columns = ["sequence", "structure", "predicted_loop_type"]



def to_seq(df):

    return np.transpose(

        np.array(

            df[sequence_columns]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )



train = to_seq(train_df)

public = to_seq(public_df)

private = to_seq(private_df)



train.shape, public.shape, private.shape
def to_one_hot(df):

    temp = np.transpose(

        np.array([

            df[col]

            .apply(lambda seq: [token2int_map[col][x] for x in seq])

            .values

            .tolist()

            for col in sequence_columns

        ]),

        (1, 2, 0)

    )

    ohe_1 = tf.keras.utils.to_categorical(temp[:,:,0], 4)

    ohe_2 = tf.keras.utils.to_categorical(temp[:,:,1], 3)

    ohe_3 = tf.keras.utils.to_categorical(temp[:,:,2], 7)

    return np.concatenate([ohe_1, ohe_2, ohe_3], axis=2)



train_ohe = to_one_hot(train_df)

public_ohe = to_one_hot(public_df)

private_ohe = to_one_hot(private_df)



train_ohe.shape, public_ohe.shape, private_ohe.shape
def get_adjacency_matrix(inps):

    As = []

    for row in range(0, inps.shape[0]):

        A = np.zeros((inps.shape[1], inps.shape[1]))

        stack = []

        opened_so_far = []



        for seqpos in range(0, inps.shape[1]):

            # A[seqpos, seqpos] = 1

            if inps[row, seqpos, 1] == 0:

                stack.append(seqpos)

                opened_so_far.append(seqpos)

            elif inps[row, seqpos, 1] == 1:

                openpos = stack.pop()

                A[openpos, seqpos] = 1

                A[seqpos, openpos] = 1

        As.append(A)

    return np.array(As)



train_adj = get_adjacency_matrix(train)

public_adj = get_adjacency_matrix(public)

private_adj = get_adjacency_matrix(private)



train_adj.shape, public_adj.shape, private_adj.shape
train_adj.mean(), public_adj.mean(), private_adj.mean()
def get_bpps(mRNA_ids):

    bpps = []

    for mRNA_id in tqdm(mRNA_ids):

        bpps.append(

            np.load(f"{bpps_path}/{mRNA_id}.npy"),

        )

    return np.array(bpps)





train_bpps = get_bpps(train_df.id.values)

public_bpps = get_bpps(public_df.id.values)

private_bpps = get_bpps(private_df.id.values)



train_bpps.shape, public_bpps.shape, private_bpps.shape 
train_bpps.mean(), public_bpps.mean(), private_bpps.mean() 
train_bpps_stats = [train_bpps.mean(axis=2), train_bpps.max(axis=2)]

public_bpps_stats = [public_bpps.mean(axis=2), public_bpps.max(axis=2)]

private_bpps_stats = [private_bpps.mean(axis=2), private_bpps.max(axis=2)]
train_bpps_stats = np.concatenate([stats[:,:,None] for stats in train_bpps_stats], axis=2)

public_bpps_stats = np.concatenate([stats[:,:,None] for stats in public_bpps_stats], axis=2)

private_bpps_stats = np.concatenate([stats[:,:,None] for stats in private_bpps_stats], axis=2)



train_bpps_stats.shape, public_bpps_stats.shape, private_bpps_stats.shape
scored_seq_length = 68



# loss functions

def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return K.sqrt(mse)



def mcrmse(y_actual, y_pred):

    score = 0

    for i in range(y_actual.shape[2]):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / y_actual.shape[2]

    return score

    



def build_model(input_seq_len=107, output_seq_len=scored_seq_length):

    

    def _bi_gru_block(x, hidden_dim, dropout):

        gru = L.Bidirectional(

            L.GRU(hidden_dim, 

                  dropout=dropout,

                  return_sequences=True,

                 ),

        )(x)

        return gru



    def _conv_block(x, adj_m, bpp_m, conv_filters, graph_channels):

        # local 1-D convolution

        conv = L.Conv1D(

            conv_filters, 5,

            padding='same',

            activation='tanh',

        )(x)

        

        # graph convolution

        gcn_1 = GraphConv(

            graph_channels,

            activation='tanh',

        )([conv, adj_m])

        

        gcn_2 = GraphConv(

            graph_channels,

            activation='tanh',

        )([conv, bpp_m])



        conv = L.Concatenate()([conv, gcn_1, gcn_2])

        conv = L.Activation("relu")(conv)

        conv = L.SpatialDropout1D(0.1)(conv)

        

        return conv

    

    # inputs

    one_hot_encoding_inputs = L.Input(shape=(input_seq_len, 14), name="onehot")

    # adjacency matrix about seq. connectivity

    adj_matrix_inputs = L.Input((input_seq_len, input_seq_len), name="adjmatrix")

    # base pair proba

    base_pair_proba_inputs = L.Input((input_seq_len, input_seq_len), name="pairproba")

    # base pair proba stats

    base_pair_proba_stats_inputs = L.Input(shape=(input_seq_len, 2), name="pairprobastats")

    

    merged_inputs = L.Concatenate()([one_hot_encoding_inputs, base_pair_proba_stats_inputs])

    

    # convolution and recurrent blocks.

    hidden = _conv_block(merged_inputs, adj_matrix_inputs, base_pair_proba_inputs, 512, 80)

    hidden = _bi_gru_block(hidden, 256, 0.5)

    hidden = _conv_block(hidden, adj_matrix_inputs, base_pair_proba_inputs, 512, 80)

    hidden = _bi_gru_block(hidden, 256, 0.5)

    

    out = hidden[:, :output_seq_len]

    out = L.Dense(5, activation='linear')(out)

    

    model = tf.keras.Model(

        inputs=[

            one_hot_encoding_inputs,

            adj_matrix_inputs,

            base_pair_proba_inputs,

            base_pair_proba_stats_inputs,

        ],

        outputs=out,

    )



    return model



model = build_model()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
split_results  = train_test_split(

    train_ohe,

    train_adj,

    train_bpps,

    train_bpps_stats,

    train_y,

    train_df.signal_to_noise,

    train_df.SN_filter,

    test_size=0.1,

    random_state=42,

)



[a.shape for a in split_results]
trn_ohe, val_ohe, trn_adj, val_adj, trn_bpps, val_bpps, trn_bpps_stats, val_bpps_stats, trn_y, val_y, trn_snr, val_snr, trn_snf, val_snf = split_results
model = build_model()

model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)
trn_inputs = [trn_ohe, trn_adj, trn_bpps, trn_bpps_stats]

val_inputs = [val_ohe, val_adj, val_bpps, val_bpps_stats]
# only validate on data with sn_filter = 1

val_mask = np.where((val_snf==1))

val_inputs = [val_input[val_mask] for val_input in val_inputs]

val_y = val_y[val_mask]
sample_weight = np.log(trn_snr+1.11)/2
history = model.fit(

    trn_inputs, trn_y,

    validation_data = (val_inputs, val_y),

    batch_size=64,

    epochs=300,

    sample_weight=sample_weight,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, monitor='val_loss'),

        tf.keras.callbacks.ModelCheckpoint(f'model.h5',save_best_only=True, verbose=0, monitor='val_loss'),

        tf.keras.callbacks.EarlyStopping(

            patience=20, 

            monitor='val_loss',

            verbose=0,

            mode="auto",

            baseline=None,

            restore_best_weights=True,

        ),

    ],

    verbose=2

)

print(f"Min validation loss history={min(history.history['val_loss'])}")
model.load_weights(f'model.h5')
val_preds = model.predict(val_inputs)

tf.reduce_mean(mcrmse(val_y, val_preds))
model_public = build_model(107, 107)

model_private = build_model(130, 130)



model_public.load_weights(f'model.h5')

model_private.load_weights(f'model.h5')
public_inputs = [public_ohe, public_adj, public_bpps, public_bpps_stats,]

private_inputs = [private_ohe, private_adj, private_bpps, private_bpps_stats,]
test_preds = [model_public.predict(public_inputs), model_private.predict(private_inputs)]

test_dfs = [public_df, private_df]



test_preds[0].shape, test_preds[1].shape
preds_ls = []

for df, preds in zip(test_dfs, test_preds):

    for i, uid in tqdm(enumerate(df.id)):

        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()



test_df.shape, preds_df.shape
submission = preds_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

submission.to_csv(f'submission.csv', index=False)

print(f'wrote to submission.csv')
submission.shape, pd.read_csv(sample_sub_path).shape, test_df.shape