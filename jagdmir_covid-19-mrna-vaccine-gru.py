import json



import pandas as pd

import numpy as np

import plotly.express as px

import tensorflow.keras.layers as L

import tensorflow as tf

from sklearn.model_selection import train_test_split
data_dir = '/kaggle/input/stanford-covid-vaccine/'

train = pd.read_json(data_dir + 'train.json', lines=True)

test = pd.read_json(data_dir + 'test.json', lines=True)

sample_df = pd.read_csv(data_dir + 'sample_submission.csv')
train.shape,test.shape
train.head()
test.head()
sample_df.head()
print("Unique values & no. of occurences for seq_scored in the training dataset:\n",train.seq_scored.value_counts())

print("\nUnique values & no. of occurences for seq_scored in the test dataset:\n",test.seq_scored.value_counts())
# training dataset

deg_columns = ['reactivity','deg_error_Mg_pH10', 'deg_error_pH10','deg_error_Mg_50C', 'deg_error_50C', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']



for col in deg_columns:

    length = []

    for each in range(train.shape[0]):

        length.append(len(train[col].iloc[each]))



    print("Length of different values for " + col + " in training dataset:",set(length))

print("Unique values & there occurences for seq_length in the training dataset:\n",train.seq_length.value_counts())

print("\nUnique values & there occurences for seq_length in the test dataset:\n",test.seq_length.value_counts())
# training dataset

length = []

for each in range(train.shape[0]):

    length.append(len(train.sequence.iloc[each]))



print("length of different values for sequence in training dataset:",set(length))



# test dataset

length = []

for each in range(test.shape[0]):

    length.append(len(test.sequence.iloc[each]))



print("\nlength of different values for sequence in test dataset:",set(length))

# training dataset

length = []

for each in range(train.shape[0]):

    length.append(len(train.structure.iloc[each]))



print("length of different values for structure in training dataset:",set(length))



# test dataset

length = []

for each in range(test.shape[0]):

    length.append(len(test.structure.iloc[each]))



print("\nlength of different values for structure in test dataset:",set(length))

# training dataset

length = []

for each in range(train.shape[0]):

    length.append(len(train.predicted_loop_type.iloc[each]))



print("length of different values for predicted_loop_type in training dataset:",set(length))



# test dataset

length = []

for each in range(test.shape[0]):

    length.append(len(test.predicted_loop_type.iloc[each]))



print("\nlength of different values for predicted_loop_type in test dataset:",set(length))

# filter records with signal to noise < 1

train = train.query("signal_to_noise >= 1")

train.shape
# This function would help us converting the target variables into an array which can be fed into keras model

def pandas_list_to_array(df):

    """

    Input: dataframe of shape (x, y), containing list of length l

    Return: np.array of shape (x, l, y)

    """

    

    return np.transpose(

        np.array(df.values.tolist()),

        (0, 2, 1)

    )
# We are defining a function here to take care of the conversion

# df would be the training or the test dataset

# token2int is dictionary which contains the character/integer mapping



def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):

    return pandas_list_to_array(

        df[cols].applymap(lambda seq: [token2int[x] for x in seq])

    )
# predictor variables

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']
# we are using a dictinoary here to map each character with a unique integer

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



# calling the function defined above to apply the actual character to integer conversion

# train_inputs is the dataframe we are going to use to feed our keras model

train_inputs = preprocess_inputs(train, token2int)



# call the function to reshape the predictor variables to convert into an array which can be fed into keras model

train_labels = pandas_list_to_array(train[pred_cols])
# sets the random seed

tf.random.set_seed(2020)

np.random.seed(2020)
# This is to generate a new set of random values every time

y_true = tf.random.normal((32, 68, 3))

y_pred = tf.random.normal((32, 68, 3))
# function to calculate average across all RMSE values for each column

def MCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
def gru_layer(hidden_dim, dropout):

    return L.Bidirectional(L.GRU(

        hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal'))
def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5, 

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
x_train, x_val, y_train, y_val = train_test_split(

    train_inputs, train_labels, test_size=.1, random_state=34, stratify=train.SN_filter)
public_df = test.query("seq_length == 107")

private_df = test.query("seq_length == 130")



public_inputs = preprocess_inputs(public_df, token2int)

private_inputs = preprocess_inputs(private_df, token2int)
model = build_model(embed_size=len(token2int))

model.summary()
history = model.fit(

    x_train, y_train,

    validation_data=(x_val, y_val),

    batch_size=32,

    epochs=50,

    verbose=2,

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(patience=5),

        tf.keras.callbacks.ModelCheckpoint('model.h5')

    ]

)

fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'MCRMSE'}, 

    title='Training History')

fig.show()
# Caveat: The prediction format requires the output to be the same length as the input,

# although it's not the case for the training data.

model_public = build_model(seq_len=107, pred_len=107, embed_size=len(token2int))

model_private = build_model(seq_len=130, pred_len=130, embed_size=len(token2int))



model_public.load_weights('model.h5')

model_private.load_weights('model.h5')
public_preds = model_public.predict(public_inputs)

private_preds = model_private.predict(private_inputs)
preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)

preds_df.head()
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)