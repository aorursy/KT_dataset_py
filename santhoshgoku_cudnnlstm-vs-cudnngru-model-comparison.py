import numpy as np

import os

import pandas as pd



#ploting libs

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

import plotly.express as px



#machine learning libs

import tensorflow as tf

from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU 

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dropout, Dense,Conv1D, MaxPooling1D, Bidirectional

from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
data_dir = '/kaggle/input/stanford-covid-vaccine/'

train = pd.read_json(data_dir + 'train.json', lines=True)

test = pd.read_json(data_dir + 'test.json', lines=True)

sample_submit = pd.read_csv(data_dir + 'sample_submission.csv')
train.head()
test.head()
sample_submit.head()
# we could see the below columns are the once to be predicted

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
#the metrics which is used to calculate the score

def MCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
# converting the pandas list to the array of size (0,2,1)

def pandas_list_to_array(df):

    """

    Input: dataframe of shape (x, y), containing list of length l

    Return: np.array of shape (x, l, y)

    """

    

    return np.transpose(

        np.array(df.values.tolist()),

        (0, 2, 1)

    )
#getting the equivalent token value for the three columns

def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):

    return pandas_list_to_array(

        df[cols].applymap(lambda seq: [token2int[x] for x in seq])

    )
# this toke2int contains the mapped dict of the charaters

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



train_inputs = preprocess_inputs(train, token2int)

train_labels = pandas_list_to_array(train[pred_cols])
# spliting up of data to train and test

x_train, x_val, y_train, y_val = train_test_split(

    train_inputs, train_labels, test_size=.1, random_state=38

)
a = train["signal_to_noise"].plot.kde(legend=True,color="#F8C471",linewidth=2.8)

plt.legend(fontsize=10)

plt.xlabel("signal to noise")
a = sns.countplot(data=train,x="SN_filter",color="blue",palette=["#F5B041","#58D68D"],dodge=False)

value = dict(train["SN_filter"].value_counts())

for index, row in value.items():

    a.text(index,row, row, color='black', ha="center")



plt.title("SN filter")
#defining the gru layer

def CUDNNGgru_layer(hidden_dim):

    return Bidirectional(

        CuDNNGRU(hidden_dim,return_sequences=True)

    )
#defining the CuDNNLSTM layer

def CuDNNlstm_layer(hidden_dim):

    return Bidirectional(

                                CuDNNLSTM(hidden_dim,

                                return_sequences=True))
# defining the model

def build_model(CuDNNlstm=True,seq_len=107, pred_len=68, dropout=0.5,embed_dim=100, hidden_dim=128):

    

    inputs = tf.keras.layers.Input(shape=(seq_len, 3))



    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    

    reshaped = tf.keras.layers.SpatialDropout1D(.2)(reshaped)

    

    if CuDNNlstm:

        hidden = CuDNNlstm_layer(hidden_dim)(reshaped)

        hidden = CuDNNlstm_layer(hidden_dim)(hidden)

        reshaped = tf.keras.layers.SpatialDropout1D(.2)(hidden)

        hidden = CuDNNlstm_layer(hidden_dim)(reshaped)

        hidden = CuDNNlstm_layer(hidden_dim)(hidden)

        reshaped = tf.keras.layers.SpatialDropout1D(.2)(hidden)

        hidden = CuDNNlstm_layer(hidden_dim)(reshaped)

        

    else:

        hidden = CUDNNGgru_layer(hidden_dim)(reshaped)

        hidden = CUDNNGgru_layer(hidden_dim)(hidden)

        reshaped = tf.keras.layers.SpatialDropout1D(.2)(hidden)

        hidden = CUDNNGgru_layer(hidden_dim)(reshaped)

        hidden = CUDNNGgru_layer(hidden_dim)(hidden)

        reshaped = tf.keras.layers.SpatialDropout1D(.2)(hidden)

        hidden = CUDNNGgru_layer(hidden_dim)(reshaped)

    

    #only making predictions on the first part of each sequence

    truncated = hidden[:, :pred_len]

    

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)



    model = tf.keras.Model(inputs=inputs, outputs=out)

    

    model.compile(optimizer = Adam(learning_rate=0.001,amsgrad=True), loss=MCRMSE)

    

    return model
#building the CUDNNlstm model

CuDNNlstm_model = build_model()

CuDNNlstm_model.summary()
#building the CUDNNGgru model

CUDNNgru_model = build_model(CuDNNlstm=False)

CUDNNgru_model.summary()
CuDNNlstm_history = CuDNNlstm_model.fit(

    x_train, y_train,

    validation_data=(x_val, y_val),

    batch_size=32,

    epochs=100,

    verbose=2,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(patience=5),

        tf.keras.callbacks.ModelCheckpoint('CuDNNlstm_model.h5')

    ]

)
CUDNNgru_history = CUDNNgru_model.fit(

    x_train, y_train,

    validation_data=(x_val, y_val),

    batch_size=64,

    epochs=100,

    verbose=2,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(patience=5),

        tf.keras.callbacks.ModelCheckpoint('CUDNNGgru_model.h5')

    ]

)
fig = px.line(

    CUDNNgru_history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'MCRMSE'}, 

    title='Training Loss -  CUDNNgru')

fig.show()
fig = px.line(

    CuDNNlstm_history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'MCRMSE'}, 

    title='Training Loss - CuDNNlstm')

fig.show()
#seperating the test set based on the length

public_df = test.query("seq_length == 107")

private_df = test.query("seq_length == 130")



public_inputs = preprocess_inputs(public_df, token2int)

private_inputs = preprocess_inputs(private_df, token2int)
#seperate model for public and private

model_public = build_model(seq_len=107, pred_len=107)

model_private = build_model(seq_len=130, pred_len=130)



model_public.load_weights('CuDNNlstm_model.h5')

model_private.load_weights('CuDNNlstm_model.h5')
#making the predictions on public and private inputs

public_preds = model_public.predict(public_inputs)

private_preds = model_private.predict(private_inputs)
# getting the predicted values

preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)

preds_df.head()
# creating the submission csv

submission = sample_submit[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)