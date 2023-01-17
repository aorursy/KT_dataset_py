import pandas as pd
import numpy as np
import json
import tensorflow.keras.layers as L
train = pd.read_json("../input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json", lines=True)
sample_df = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
train.head()
!pip install datasist
import datasist as ds
ds.structdata.check_train_test_set(train, test, index=None, col=None)
# ds.structdata.describe(train)
train.info()
train.describe()
import missingno as msno
msno.matrix(train)
msno.bar(train)
pd.set_option('max_columns', 100)
train
len(train['structure'][1])
length = []
for struct in train['structure']:
    length.append(len(struct))
length
train["flag"] = "train"
test["flag"] = "test"
# !pip install datasist
import datasist as ds
all_data, ntrain, ntest = ds.structdata.join_train_and_test(train, test)
# #later splitting after transformations
# train_new = all_data[:ntrain]
# test_new = all_data[ntrain:]
all_data
count = 0
listof = []
for data in all_data['predicted_loop_type']:
    for letter in str(data):
        if letter == "S":
            count += 1
#     listof.append(count)
all_data["S"] = all_data['predicted_loop_type'].str.count("S")
all_data["M"] = all_data['predicted_loop_type'].str.count("M")
all_data["I"] = all_data['predicted_loop_type'].str.count("I")
all_data["B"] = all_data['predicted_loop_type'].str.count("B")
all_data["H"] = all_data['predicted_loop_type'].str.count("H")
all_data["X"] = all_data['predicted_loop_type'].str.count("X")
all_data
train = all_data[:ntrain]
test = all_data[ntrain:]
# train['S'] = 
train['predicted_loop_type']
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
train[pred_cols]
y = train[pred_cols]
y = np.array(train[train.signal_to_noise > 1][pred_cols].values.tolist()).transpose((0, 2, 1))
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
X = preprocess_inputs(train[train.signal_to_noise > 1])
X
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
test.shape
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    inputs = L.Input(shape=(seq_len, 3))

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    hidden = gru_layer(hidden_dim, dropout)(reshaped)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    # Since we are only making predictions on the first part of each sequence, we have
    # to truncate it
    truncated = hidden[:, :pred_len]
    out1 = L.BatchNormalization()(truncated)
    out = L.Dense(5, activation='linear')(out1)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss='mse')
    
    return model
model = build_model()
model.summary()
history = model.fit(
    X, y, 
    batch_size=64,
    epochs=150,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint('model.h5')
    ],
    validation_split=0.25
)
# import pandas as pd
# loss = pd.DataFrame({loss: model.history.history["loss"], acc: model.history.history["val_loss"] })
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)
# although it's not the case for the training data.
model_short = build_model(seq_len=107, pred_len=107)
model_long = build_model(seq_len=130, pred_len=130)

model_short.load_weights('model.h5')
model_long.load_weights('model.h5')

public_preds = model_short.predict(public_inputs)
private_preds = model_long.predict(private_inputs)
preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission1234.csv', index=False)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize = (20, 10))

ax[0].plot(history.history['loss'])
ax[0].plot(history.history['val_loss'])


ax[0].set_title('GRU')

ax[0].legend(['train', 'validation'], loc = 'upper right')

ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')