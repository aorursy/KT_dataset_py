# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import math



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import tensorflow as tf

from tensorflow import keras



from learntools.core import binder; binder.bind(globals())

from learntools.embeddings.ex1_embedding_layers import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



input_dir = '../input/elo-merchant-category-recommendation'



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/elo-merchant-category-lab/trans_by_card.csv")
df.head()
fields = ['card_id', 'authorized_flag', 'merchant_id', 'purchase_amount']

df = pd.read_csv(os.path.join(input_dir, 'historical_transactions.csv'), usecols=fields)
df = df[df['authorized_flag'] == 'Y']
df.head()
df = df.groupby(['card_id', 'merchant_id']).agg({'authorized_flag':'count',

                                                 'purchase_amount':'sum'

                                                }).reset_index()
#plt.hist(df.purchase_amount, bins=1000)
merchant_ids = set(df.merchant_id)

merchant_dict = dict()

i = 0

for s in merchant_ids:

    merchant_dict[s] = i

    i +=1

df['merchant_id_numeric'] = df['merchant_id'].apply(lambda x: merchant_dict[x])
card_ids = set(df.card_id)

card_dict = dict()

i = 0

for s in card_ids:

    card_dict[s] = i

    i += 1

df['card_id_numeric'] = df['card_id'].apply(lambda x: card_dict[x])
del [card_dict, card_ids, merchant_ids, merchant_dict]
#df.head()
from sklearn import preprocessing



x = np.array(df.purchase_amount) #returns a numpy array

standard_scaler = preprocessing.StandardScaler()

x_scaled = standard_scaler.fit_transform(x.reshape(-1,1))

df['y'] = x_scaled
# Some hyperparameters. (You might want to play with these later)

LR = .005 # Learning rate

EPOCHS = 8 # Default number of training epochs (i.e. cycles through the training data)

hidden_units = (32,4) # Size of our hidden layers



def build_and_train_model(merchant_embedding_size=8, card_embedding_size=8, verbose=2, epochs=EPOCHS):

    tf.set_random_seed(1); np.random.seed(1); random.seed(1) # Set seeds for reproducibility

    card_id_input = keras.Input(shape=(1,), name='card_id_numeric')

    merchant_id_input = keras.Input(shape=(1,), name='merchant_id_numeric')

    card_embedded = keras.layers.Embedding(df.card_id_numeric.max()+1, card_embedding_size, 

                                       input_length=1, name='card_embedding')(card_id_input)

    merchant_embedded = keras.layers.Embedding(df.merchant_id_numeric.max()+1, merchant_embedding_size, 

                                        input_length=1, name='merchant_embedding')(merchant_id_input)

    bias_embedded = keras.layers.Embedding(df.merchant_id_numeric.max()+1, 1, input_length=1, name='bias',

                                      )(merchant_id_input)

    concatenated = keras.layers.Concatenate()([card_embedded, merchant_embedded])

    out = keras.layers.Flatten()(concatenated)



    # Add one or more hidden layers

    for n_hidden in hidden_units:

        out = keras.layers.Dense(n_hidden, activation='relu')(out)



    # A single output: our predicted rating

    out = keras.layers.Dense(1, activation='linear', name='prediction')(out)

    

    merchant_bias = keras.layers.Flatten()(bias_embedded)



    out = keras.layers.Add()([out, merchant_bias])



    model = keras.Model(

        inputs = [card_id_input, merchant_id_input],

        outputs = out,

    )

    model.summary()

    

    model.compile(

        tf.train.AdamOptimizer(LR),

        loss='MSE',

        metrics=['MAE'],

    )

    history = model.fit(

        [df.card_id_numeric, df.merchant_id_numeric],

        df.y,

        batch_size=5 * 10**3,

        epochs=epochs,

        verbose=verbose,

        validation_split=.15,

    )

    return history
history = build_and_train_model(verbose=1, card_embedding_size=64, merchant_embedding_size=64, epochs=3)
history_FS = (15, 5)

def plot_history(histories, keys=('mean_absolute_error',), train=True, figsize=history_FS):

    if isinstance(histories, tf.keras.callbacks.History):

        histories = [ ('', histories) ]

    for key in keys:

        plt.figure(figsize=history_FS)

        for name, history in histories:

            val = plt.plot(history.epoch, history.history['val_'+key],

                           '--', label=str(name).title()+' Val')

            if train:

                plt.plot(history.epoch, history.history[key], color=val[0].get_color(), alpha=.5,

                         label=str(name).title()+' Train')



        plt.xlabel('Epochs')

        plt.ylabel(key.replace('_',' ').title())

        plt.legend()

        plt.title(key)



        plt.xlim([0,max(max(history.epoch) for (_, history) in histories)])

plot_history([ 

    ('base model', history),

])