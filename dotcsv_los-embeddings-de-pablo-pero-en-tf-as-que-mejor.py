# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
battles = pd.read_csv('../input/battles.csv')

pokemon = pd.read_csv('../input/pokemon.csv')

test    = pd.read_csv('../input/test.csv')
print(battles.shape)

battles.head()
print(pokemon.shape)

pokemon.head()
print(model.summary())
import tensorflow as tf

from tensorflow.keras import Sequential, Model

from tensorflow.keras.layers import Embedding, Dense, Input

from tensorflow.keras.optimizers import SGD



from sklearn.model_selection import train_test_split



# train, test = train_test_split(battles, test_size=0.0)

train = battles



embed_size = 128



# Creamos los inputs para cada poke-id

poke1 = Input(shape=[1], name="input_1")

poke2 = Input(shape=[1], name="input_2")



# Embedding layer como capa inicial.

embed = Embedding(800, embed_size, input_length=1)



embe1 = embed(poke1)

embe2 = embed(poke2)



# Prueba concatenando y substrayendo los embeddings.

conct = tf.keras.layers.concatenate([embe1, embe2])

# conct = tf.keras.layers.subtract([embe1, embe2])



o = tf.keras.layers.Flatten()(Dense(1, activation='sigmoid')(conct))

 

model = Model(inputs=[poke1, poke2], outputs=o)

model.compile(optimizer=SGD(lr=0.5), loss='mse', metrics=['accuracy'])



model.fit([train.First_pokemon - 1, train.Second_pokemon - 1], train.Winner, batch_size=128, epochs=2000, validation_split=0.1)
submission = test.iloc[:, 0:1]

submission['Winner'] = (model.predict([test.First_pokemon - 1, test.Second_pokemon - 1]) > 0.5).astype(int)

submission.head(100)

submission.to_csv('submission.csv', index=False)
submission