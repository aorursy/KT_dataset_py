import json

import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
def LOSS_MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(
        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )

def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )
data_dir = '/kaggle/input/stanford-covid-vaccine/'
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))


train = pd.read_json(data_dir + 'train.json', lines=True)
test = pd.read_json(data_dir + 'test.json', lines=True)
sample_df = pd.read_csv(data_dir + 'sample_submission.csv')

train = train.query("signal_to_noise >= 1")

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols])

x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=.1, random_state=34, stratify=train.SN_filter)

public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

public_inputs = preprocess_inputs(public_df, token2int)
private_inputs = preprocess_inputs(private_df, token2int)
def build_model_structure(embed_size=14, seq_len=107, pred_len=68, dropout=0.5, 
                sp_dropout=0.2, embed_dim=200, hidden_dim=256, n_layers=3):
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
    model.compile(tf.optimizers.Adam(), loss=MCRMSE)
    
    return model
keras.backend.clear_session()
weight_file='/kaggle/input/learned-model/model.h5'
#public model
pre_trained_public_model = build_model_structure(seq_len=107, pred_len=107)
pre_trained_public_model.load_weights(weight_file)


#private model
pre_trained_private_model = build_model_structure(seq_len=130, pred_len=130)
pre_trained_private_model.load_weights(weight_file)
# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_public_model.layers:
    layer.trainable=False

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_private_model.layers:
    layer.trainable=False
       
# Get the summary
pre_trained_public_model.summary()
# Select the last layer
pred_len=68
last_layer = pre_trained_public_model.get_layer('dense')
last_output = last_layer.output


# Add a fully connected layer with 1,024 hidden units and ReLU activation
new_layer = lstm_layer(200, dropout=.4)(last_output)
#x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
new_layer = layers.Dropout(.2)(new_layer)    

truncated = new_layer[:, :pred_len]

out = tf.keras.layers.Dense(5, activation='linear')(truncated)
transferred_model = tf.keras.Model(inputs=pre_trained_public_model.input,outputs=out)
transferred_model.compile(tf.optimizers.Adam(), loss=MCRMSE)
pre_trained_private_model.summary()
# Select the last layer
pred_len=107
last_layer = pre_trained_private_model.get_layer('dense_1')
last_output = last_layer.output


# Add a fully connected layer with 1,024 hidden units and ReLU activation
new_layer = lstm_layer(200, dropout=.4)(last_output)
#x = layers.Dense(1024, activation='relu')(x)

# Add a dropout rate of 0.2
new_layer = layers.Dropout(.2)(new_layer)    

truncated = new_layer[:, :pred_len]

out = tf.keras.layers.Dense(5, activation='linear')(truncated)
transferred_pr_model = tf.keras.Model(inputs=pre_trained_private_model.input,outputs=out)
transferred_pr_model.compile(tf.optimizers.Adam(), loss=MCRMSE)
history = transferred_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=40,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint('transferred_model.h5')
    ]
)
fig = px.line(
    history.history, y=['loss', 'val_loss'],
    labels={'index': 'epoch', 'value': 'MCRMSE'}, 
    title='Training History')
fig.show()
transferred_model_public = transferred_model
transferred_model_private = transferred_pr_model

transferred_model_public.load_weights('transferred_model.h5')
transferred_model_private.load_weights('transferred_model.h5')

public_preds = transferred_model_public.predict(public_inputs)
private_preds = transferred_model_private.predict(private_inputs)

preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.head()
preds_df.to_csv('preds_df.csv', index=False)