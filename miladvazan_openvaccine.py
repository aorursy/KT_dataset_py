import pandas as pd, numpy as np

import math, json, gc, random, os, sys

from matplotlib import pyplot as plt

from tqdm import tqdm

import tensorflow as tf

import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

tf.random.set_seed(256)

np.random.seed(256)
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
train.head()
test.head()
print('train',train.shape)

print('test',test.shape)
target = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
t2t = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    return np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [t2t[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )
train_inputs = preprocess_inputs(train.loc[train.SN_filter == 1])

train_labels = np.array(train.loc[train.SN_filter == 1][target].values.tolist()).transpose((0, 2, 1))
x_train, x_val, y_train, y_val = train_test_split(

    train_inputs, train_labels, test_size=.1, random_state=256

)
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df)

private_inputs = preprocess_inputs(private_df)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(

        L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal')

    )
def GRU(embed_size, seq_len=107, pred_len=68, dropout=0.5, 

                sp_dropout=0.5, embed_dim=75, hidden_dim=128, n_layers=2):

    inputs = L.Input(shape=(seq_len, 3))

    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)

    

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])

    )

    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)

    

    for x in range(n_layers):

        hidden = gru_layer(hidden_dim, dropout)(hidden)

    

    # Since we are only making predictions on the first part of each sequence, 

    # we have to truncate it

    truncated = hidden[:, :pred_len]

    out = L.Dense(5, activation='linear')(truncated)

    

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.optimizers.Adam(), loss='mse')

    

    return model
model = GRU(embed_size=len(t2t))

model.summary()
gru_f = []

gru_private_preds  = np.zeros((private_df.shape[0], 130, 5))

gru_public_preds = np.zeros((public_df.shape[0], 107, 5))

kfold = KFold(5, shuffle = True, random_state = 128)



for f, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

    

    sv_gru = tf.keras.callbacks.ModelCheckpoint(f'gru-{f}.h5')

    

    train_ = train_inputs[train_index]

    train_labs = train_labels[train_index]

    val_ = train_inputs[val_index]

    val_labs = train_labels[val_index]

        

    gru = GRU(embed_size=len(t2t))

    history = gru.fit(

                        train_, train_labs, 

                        validation_data=(val_,val_labs),

                        batch_size=120,

                        epochs=100,

                        callbacks=[lr_callback,sv_gru],

                        verbose = 2)  

    

    gru_f.append(history)

    

    gru_short = GRU( embed_size=len(t2t),seq_len=107, pred_len=107)

    gru_short.load_weights(f'gru-{f}.h5')

    gru_public= gru_short.predict(public_inputs) / 5

    

    gru_long = GRU(embed_size=len(t2t), seq_len=130, pred_len=130)

    gru_long.load_weights(f'gru-{f}.h5')

    gru_private = gru_long.predict(private_inputs) / 5

    

    gru_public_preds += gru_public

    gru_private_preds += gru_private

    

    del gru_short, gru_long
print(f" GRU mean fold validation loss: {np.mean([min(history.history['val_loss']) for history in gru_f])}")
preds_gru = []



for df, preds in [(public_df,  gru_public_preds), (private_df, gru_private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=target)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_gru.append(single_df)



preds_gru_df = pd.concat(preds_gru)

preds_gru_df.head()
submission = sample_sub[['id_seqpos']].merge(preds_gru_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)