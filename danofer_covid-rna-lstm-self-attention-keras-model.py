import pandas as pd

import numpy as np

import json

import tensorflow.keras.layers as L

import tensorflow as tf

import plotly.express as px
# This will tell us the columns we are predicting

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
### Keras self attention example code (not formatted for this target shape!! - won't work without that  

### https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



"""

## Implement multi head self attention as a Keras layer

"""





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





"""

## Implement a Transformer block as a layer

"""





class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

        super(TransformerBlock, self).__init__()

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        self.ffn = keras.Sequential(

            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]

        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)

        self.dropout2 = layers.Dropout(rate)



    def call(self, inputs, training):

        attn_output = self.att(inputs)

        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)





"""

## Implement embedding layer

Two seperate embedding layers, one for tokens, one for token index (positions).

"""





class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim):

        super(TokenAndPositionEmbedding, self).__init__()

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)



    def call(self, x):

        maxlen = tf.shape(x)[-1]

        positions = tf.range(start=0, limit=maxlen, delta=1)

        positions = self.pos_emb(positions)

        x = self.token_emb(x)

        return x + positions

from tensorflow.keras.layers import  Layer

from tensorflow.keras.layers import *

from tensorflow.keras.models import *

from tensorflow.keras import backend as K



### https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137

#### Built to receive 3D tensors and output 3D tensors (return_sequences=True) or 2D tensors (return_sequences=False)

class attention(Layer):

    

    def __init__(self, return_sequences=True):

        self.return_sequences = return_sequences

        super(attention,self).__init__()

        

    def build(self, input_shape):

        

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),

                               initializer="normal")

        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),

                               initializer="zeros")

        

        super(attention,self).build(input_shape)

        

    def call(self, x):

        

        e = K.tanh(K.dot(x,self.W)+self.b)

        a = K.softmax(e, axis=1)

        output = x*a

        

        if self.return_sequences:

            return output

        

        return K.sum(output, axis=1)

    

    ### https://stackoverflow.com/questions/58678836/notimplementederror-layers-with-arguments-in-init-must-override-get-conf

    ### I'm not doing this right... 

    ### https://www.tensorflow.org/guide/keras/save_and_serialize

    def get_config(self):



        config = super().get_config().copy()

        config.update({

#             'vocab_size': self.vocab_size,

            'num_layers': self.num_layers,

            'units': self.units,

            'd_model': self.d_model,

            'num_heads': self.num_heads,

            'dropout': self.dropout,

        })

        return config

def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))



## consider adding convlstm layer(s) - https://keras.io/api/layers/recurrent_layers/conv_lstm2d/



## example of adding attention to an lstm

# https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137

# https://stackoverflow.com/questions/56946995/how-to-build-a-attention-model-with-keras

def build_model(seq_len=107, pred_len=68, dropout=0.35, embed_dim=100, hidden_dim=128):

    inputs = L.Input(shape=(seq_len, 3))



    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))



### maybe use stackedRNN cells layer? https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells

    hidden = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(reshaped)

    hidden = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True))(hidden)



    

    # Since we are only making predictions on the first part of each sequence, we have

    # to truncate it

    truncated = hidden[:, :pred_len]

    

    out = L.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)



    model.compile(tf.keras.optimizers.Adam(), loss='mse')

    

    return model





## adding self attention - currently I have bugs, doesn' work. TODO. 

## consider adding convlstm layer(s) - https://keras.io/api/layers/recurrent_layers/conv_lstm2d/



## example of adding attention to an lstm - BUT! Tensorflow's layers are NOT self attention!

# https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137

# https://stackoverflow.com/questions/56946995/how-to-build-a-attention-model-with-keras

## another way of attention with lstm (doesn't use TF layers) - https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb



## Keras self attention example code (not formatted for this target)  -https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py

def build_model_att(seq_len=107, pred_len=68, dropout=0.45, embed_dim=100, hidden_dim=256):

    inputs = L.Input(shape=(seq_len, 3))

    

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

    

    embed = tf.keras.layers.SpatialDropout2D(rate=dropout/3)(embed) # add spatial dropout

    

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

### maybe use stackedRNN cells layer? https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells

    hidden = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout,recurrent_dropout=dropout/3, return_sequences=True))(reshaped)

    hidden = attention(return_sequences=True)(hidden)

    hidden = L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout,recurrent_dropout=dropout/3, return_sequences=True))(hidden)



    

#     hidden_att = attention(return_sequences=False)(hidden) ## https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137

#     hidden_att = tf.keras.layers.GlobalAveragePooling1D()(hidden_att)





    # Since we are only making predictions on the first part of each sequence, we have

    # to truncate it 

    ### ??? - Dan - how is this truncation ok, vs leaving the cols in ? 

    truncated = hidden[:, :pred_len]

    

# # #     # add self attention ? 

#     truncated = attention(return_sequences=True)(truncated)

    

    """

    ### Try adding TF attention - Dan: 

    #### https://www.tensorflow.org/api_docs/python/tf/keras/layers/AdditiveAttention

    # Query-value attention of shape [batch_size, Tq, filters].

#     query_value_attention_seq = tf.keras.layers.AdditiveAttention()(

#     [query_seq_encoding, value_seq_encoding])

    query_value_attention_seq = tf.keras.layers.AdditiveAttention()(

    [hidden, embed])

    # Reduce over the sequence axis to produce encodings of shape # [batch_size, filters].

    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(

    query_value_attention_seq)

    

    ### concat - Dan: 

    # Concatenate query and document encodings to produce a DNN input layer.

    joint = tf.keras.layers.Concatenate()(

    [truncated, query_value_attention])

    """

    

#     ### concat - Dan: 

#     # Concatenate query and document encodings to produce a DNN input layer.

#     joint = tf.keras.layers.Concatenate()(

#     [truncated, hidden_att])



    

    out = L.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)



    model.compile(tf.keras.optimizers.Adam(), loss='mse')

    

    return model





model = build_model_att()

model.summary()
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
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
train_inputs = preprocess_inputs(train)

train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
# model = build_model()

model = build_model_att()

model.summary()
history = model.fit(

    train_inputs, train_labels, 

    batch_size=64,

    epochs=100,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(),

#         tf.keras.callbacks.ModelCheckpoint('model.h5'), ## error with model saving/get confiug

        tf.keras.callbacks.EarlyStopping(patience=4, mode='auto',restore_best_weights=True)

    ],

    validation_split=0.3 # 0.3 ## note - validation (randomly) is unstable, and val_loss is lower when using larger validation split

)
N_EPOCHS = len(history.history['loss'])

print("number of training epochs picked:",N_EPOCHS)
fig = px.line(

    history.history, y=['loss', 'val_loss'], 

    labels={'index': 'epoch', 'value': 'Mean Squared Error'}, 

    title='Training History')

fig.show()
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
# Caveat: The prediction format requires the output to be the same length as the input,

# although it's not the case for the training data.

model_short = build_model(seq_len=107, pred_len=107)

model_long = build_model(seq_len=130, pred_len=130)





#### add workaround instead of saving model? 

model_short.fit(

    train_inputs, train_labels, 

    batch_size=64,

    epochs=N_EPOCHS,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)



model_long.fit(

    train_inputs, train_labels, 

    batch_size=64,

    epochs=N_EPOCHS,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)
# model_short.load_weights('model.h5')

# model_long.load_weights('model.h5')



public_preds = model_short.predict(public_inputs)

private_preds = model_long.predict(private_inputs)
print(public_preds.shape, private_preds.shape)
preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)