import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow import keras
train_df = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test_df = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
test_public_df = test_df[test_df['seq_length']==107]

test_private_df = test_df[test_df['seq_length']==130]



dfs = [train_df, test_public_df, test_private_df]

output_lens = [68, 107, 130]
input_cols = ['sequence', 'structure', 'predicted_loop_type']

output_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
#constants

SLIDING_SIZE = 15
# 0,1,2 represents the empty part of the window that will be added 

token2int = { char:index for index, char in enumerate('012().ACGUBEHIMSX')}
# add leading and trailing zeros to the list 

# you want for each char to have a window above

# as it is not possible for the first and last chars you add "empty" values

for df in dfs:

    for col in input_cols:

        if col == 'sequence':

            char = '0'

        if col == 'structure':

            char = '1'

        if col == 'predicted_loop_type':

            char = '2'

        print(col)

        df.loc[:, col] = df.loc[:, col].apply(lambda sequence: char * SLIDING_SIZE + sequence + char * SLIDING_SIZE)

#encode the data

seqs = []

for df in dfs:

    seq = df[input_cols].applymap(lambda sequence: [token2int[char] for char in sequence])

    seqs.append(seq)

    
def create_onehot(x):

    one_hot = np.zeros((len(x), len(token2int)))

    one_hot[np.arange(x.size), x] = 1

    return one_hot

pictures_list = []



for seq in seqs:

    sequence = np.array(seq.values.tolist())

    one_hot = np.apply_along_axis(create_onehot, 2, sequence)

    pictures = np.sum(one_hot, axis=1)

    pictures_list.append(pictures)
height = SLIDING_SIZE*2+1



slides = []



for length in output_lens:

    a = np.expand_dims(np.arange(height), axis=0)

    b = np.expand_dims(np.arange(length), axis=0).T

    slide = (a + b)

    slides.append(slide)

X = pictures_list[0][:, slides[0]]

X_test_public = pictures_list[1][:, slides[1]]

X_test_private = pictures_list[2][:, slides[2]]
X = X.reshape(-1, X.shape[2], X.shape[3], 1)

X_test_public = X_test_public.reshape(-1, X_test_public.shape[2], X_test_public.shape[3], 1)

X_test_private = X_test_private.reshape(-1, X_test_private.shape[2], X_test_private.shape[3], 1)
X.shape
y = np.array(train_df[output_cols].values.tolist())
y = np.transpose(y, (0, 2, 1))
y = y.reshape(-1, y.shape[2])
#FINALLY IMPLEMENT CNN



cnn_model = keras.models.Sequential([

    keras.layers.Input(shape=(SLIDING_SIZE*2+1, 17, 1)),

    keras.layers.Conv2D(32, kernel_size=(2,2), activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.AveragePooling2D(pool_size=(2,2)),

    keras.layers.Conv2D(64, kernel_size=(4,2), activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.AveragePooling2D(pool_size=(2,2)),

    keras.layers.Conv2D(128, kernel_size=(2,2), activation='relu'),

    keras.layers.AveragePooling2D(pool_size=(2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(100, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(50, activation='relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(5, activation='linear')

])

checkpoint_cb = keras.callbacks.ModelCheckpoint('01_cnn_model.h5', save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)



optimizer = keras.optimizers.Adam(

    learning_rate=0.001

)

cnn_model.compile(optimizer=optimizer, loss='mse')
history = cnn_model.fit(X, y, validation_split=0.25, batch_size=128, epochs=100, callbacks=[checkpoint_cb, early_stopping_cb])
cnn_model = keras.models.load_model('01_cnn_model.h5')
cnn_model.evaluate(X, y)
model = cnn_model
pred_public = model.predict(X_test_public)

pred_private = model.predict(X_test_private)
del X

del X_test_public

del X_test_private
pred_public = pred_public.reshape(-1, 107, 5)

pred_private = pred_private.reshape(-1, 130, 5)
pred_public.shape
pred_dfs = []

for ids, preds in [(test_public_df['id'], pred_public), (test_private_df['id'], pred_private)]:

    for i, id in enumerate(ids):

        pred = preds[i]

        

        df = pd.DataFrame(pred, columns=output_cols)

        df['id_seqpos'] = [f'{id}_{x}' for x in range(df.shape[0])]

        pred_dfs.append(df)



pred_df = pd.concat(pred_dfs)
pred_df
submission = sample_df[['id_seqpos']].merge(pred_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)