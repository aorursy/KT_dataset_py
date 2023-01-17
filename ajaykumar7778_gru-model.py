import pandas as pd

import numpy as np

import json

import tensorflow.keras.layers as L

import tensorflow as tf

import plotly.express as px
# This will tell us the columns we are predicting

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))



def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):

    inputs = L.Input(shape=(seq_len, 3))



    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))



    hidden = gru_layer(hidden_dim, dropout)(reshaped)

    hidden = gru_layer(hidden_dim, dropout)(hidden)

    hidden = gru_layer(hidden_dim, dropout)(hidden)

    

    # Since we are only making predictions on the first part of each sequence, we have

    # to truncate it

    truncated = hidden[:, :pred_len]

    

    out = L.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)



    model.compile(tf.keras.optimizers.Adam(), loss='mse')

    

    return model
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
model = build_model()

model.summary()
# history = model.fit(

#     train_inputs, train_labels, 

#     batch_size=64,

#     epochs=100,

#     callbacks=[

#         tf.keras.callbacks.ReduceLROnPlateau(),

#         tf.keras.callbacks.ModelCheckpoint('model.h5')

#     ],

#     validation_split=0.05

# )
# fig = px.line(

#     history.history, y=['loss', 'val_loss'], 

#     labels={'index': 'epoch', 'value': 'Mean Squared Error'}, 

#     title='Training History')

# fig.show()
# public_df = test.query("seq_length == 107").copy()

# private_df = test.query("seq_length == 130").copy()



# public_inputs = preprocess_inputs(public_df)

# private_inputs = preprocess_inputs(private_df)
# # Caveat: The prediction format requires the output to be the same length as the input,

# # although it's not the case for the training data.

# model_short = build_model(seq_len=107, pred_len=107)

# model_long = build_model(seq_len=130, pred_len=130)



# model_short.load_weights('model.h5')

# model_long.load_weights('model.h5')



# public_preds = model_short.predict(public_inputs)

# private_preds = model_long.predict(private_inputs)
# print(public_preds.shape, private_preds.shape)
# preds_ls = []



# for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

#     for i, uid in enumerate(df.id):

#         single_pred = preds[i]



#         single_df = pd.DataFrame(single_pred, columns=pred_cols)

#         single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



#         preds_ls.append(single_df)



# preds_df = pd.concat(preds_ls)
# submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

# submission.to_csv('submission.csv', index=False)