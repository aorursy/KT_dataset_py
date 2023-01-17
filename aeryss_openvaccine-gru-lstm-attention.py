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
train["sequence"][0], train["structure"][0]
train["sequence"] = train["sequence"] + "0" * 23

train["structure"] += "0" * 23

train["predicted_loop_type"] += "0" * 23
test
struct = []

seq = []

plt = []

for idx, val in test.iterrows():

    if val["seq_length"] == 107:

        struct.append(val["structure"] + "0" * 23)

        seq.append(val["sequence"] + "0" * 23)

        plt.append(val["predicted_loop_type"] + "0" * 23)

    else:

        struct.append(val["structure"])

        seq.append(val["sequence"])

        plt.append(val["predicted_loop_type"])

        

test["sequence"] = seq

test["structure"] = struct

test["predicted_loop_type"] = plt

test
#target columns

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX0')}
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
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):

    def __init__(self, embed_dim, num_heads=8):

        super(MultiHeadSelfAttention, self).__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        if embed_dim % num_heads != 0:

            raise ValueError(

                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"

            )

        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)

        self.key_dense = layers.Dense(embed_dim)

        self.value_dense = layers.Dense(embed_dim)

        self.combine_heads = layers.Dense(embed_dim)

    def get_config(self):

        config = super().get_config().copy()

        config.update({

            "embed_dim": self.embed_dim,

            "num_heads": self.num_heads,

            "projection_dim": self.projection_dim,

            "query_dense" : self.query_dense,

            "key_dense" : self.key_dense,

            "value_dense" : self.value_dense,

            "conbine_heads" : self.combine_heads

        })

        return config



    def attention(self, query, key, value):

        score = tf.matmul(query, key, transpose_b=True)

        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)

        scaled_score = score / tf.math.sqrt(dim_key)

        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)

        return output, weights



    def separate_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

        return tf.transpose(x, perm=[0, 2, 1, 3])



    def call(self, inputs):

        # x.shape = [batch_size, seq_len, embedding_dim]

        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)

        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)

        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(

            query, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        key = self.separate_heads(

            key, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        value = self.separate_heads(

            value, batch_size

        )  # (batch_size, num_heads, seq_len, projection_dim)

        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(

            attention, perm=[0, 2, 1, 3]

        )  # (batch_size, seq_len, num_heads, projection_dim)

        concat_attention = tf.reshape(

            attention, (batch_size, -1, self.embed_dim)

        )  # (batch_size, seq_len, embed_dim)

        output = self.combine_heads(

            concat_attention

        )  # (batch_size, seq_len, embed_dim)

        return output

    

class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):

        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):

        config = super().get_config().copy()

        config.update({

            "token_emb" : self.token_emb,

            "pos_emd" : self.pos_emb

        })

        return config



    def call(self, x):

        maxlen = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=maxlen, delta=1)

        positions = self.pos_emb(positions)

        x = self.token_emb(x)

        return x + positions
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



def build_model(gru=False,seq_len=130, pred_len=68, dropout=0.4,

                embed_dim=100, hidden_dim=168, atten_head = 5):

    

    x = tf.keras.layers.Input(shape=(seq_len, 3))

    tfkl = tf.keras.layers

#     x0 = tfkl.Lambda(lambda x : x[..., 0])(x)

#     x1 = tfkl.Lambda(lambda x : x[..., 1])(x)

#     x2 = tfkl.Lambda(lambda x : x[..., 2])(x)

    

#     x0 = TokenAndPositionEmbedding(seq_len, len(token2int), embed_dim)(x0)

#     x1 = TokenAndPositionEmbedding(seq_len, len(token2int), embed_dim)(x1)

#     x2 = TokenAndPositionEmbedding(seq_len, len(token2int), embed_dim)(x2)

    x0 = TokenAndPositionEmbedding(seq_len, len(token2int), embed_dim)(x)

    

#     reshaped0 = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(x0)

# #     reshaped0 = tf.reshape(

# #         embed0, shape=(-1, embed0.shape[1],  embed0.shape[2] * embed0.shape[3]))

    

#     reshaped1 = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(x1)

# #     reshaped1 = tf.reshape(

# #         embed1, shape=(-1, embed1.shape[1],  embed1.shape[2] * embed1.shape[3]))

    

#     reshaped2 = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(x2)

# #     reshaped2 = tf.reshape(

# #         embed2, shape=(-1, embed2.shape[1],  embed2.shape[2] * embed2.shape[3]))

    

#     atten = []

#     for i in range(atten_head):

#         att1 = tf.keras.layers.Attention()([reshaped0, reshaped1])

#         att2 = tfkl.Attention()([reshaped1, reshaped2])

#         att3 = tfkl.Attention()([reshaped0, reshaped2])

#         atten.append(att1)

#         atten.append(att2)

#         atten.append(att3)

    att = MultiHeadSelfAttention(32)(x0)

#     att2 = MultiHeadSelfAttention(32)(x)

#     att3 = MultiHeadSelfAttention(32)(x2)

        

#     o = tfkl.Concatenate()([att1, att2, att3])

    o = tfkl.LayerNormalization()(att)

    





#     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

#     reshaped = tf.reshape(

#         embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    

    reshaped = tf.keras.layers.SpatialDropout1D(.2)(o)

    

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



    model = tf.keras.Model(inputs=x, outputs=out)



    #some optimizers

    adam = tf.optimizers.Adam()

    radam = tfa.optimizers.RectifiedAdam()

    lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)

    ranger = tfa.optimizers.Lookahead(radam, sync_period=6)

    

    model.compile(optimizer = ranger, loss=MCRMSE)

    

    return model
train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels,

                                                                     test_size=.1, random_state=34)
if tf.config.list_physical_devices('GPU') is not None:

    print('Training on GPU')
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
gru = build_model(gru=True)

sv_gru = tf.keras.callbacks.ModelCheckpoint('model_gru.h5', save_best_only=True)



history_gru = gru.fit(

    train_inputs, train_labels, 

    validation_data=(val_inputs,val_labels),

#     batch_size=64,

    epochs=150,

    callbacks=[lr_callback,sv_gru],

    verbose = 2

)



print(f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}")
lstm = build_model(gru=False)

sv_lstm = tf.keras.callbacks.ModelCheckpoint('model_lstm.h5', save_best_only=True)



history_lstm = lstm.fit(

    train_inputs, train_labels, 

    validation_data=(val_inputs,val_labels),

#     batch_size=64,

    epochs=150,

    callbacks=[lr_callback,sv_lstm],

    verbose = 2

)



print(f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}")
import matplotlib



fig, ax = matplotlib.pyplot.subplots(1, 2, figsize = (20, 10))



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

gru_short = build_model(gru=True, seq_len=130, pred_len=107)

gru_long = build_model(gru=True, seq_len=130, pred_len=130)

lstm_short = build_model(gru=False, seq_len=130, pred_len=107)

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
# !kaggle competitions submit -c stanford-covid-vaccine -f submission.csv -m "Message"