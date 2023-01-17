import pandas as pd

import numpy as np

import tensorflow as tf



np.set_printoptions(threshold=np.inf)



inpcols=['sequence', 'structure', 'predicted_loop_type']

outcols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']



train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
print(train[inpcols].head(1))
def cust_vectorizer(inp, characters='AUCG.()ESHBXIM'):

    return [[0 if char != letter else 1 for char in characters] for letter in inp]



mapped_num= {x:i for i, x in enumerate('AUCG.()ESHBXIM')}

mapped_OHE=dict(zip(list('AUCG.()ESHBXIM'), zip(*cust_vectorizer('AUCG.()ESHBXIM'))))



print('Simple integer Mapping:\n', mapped_num,'\n\nOHE mapping:\n',mapped_OHE)
train=np.array(train[inpcols].applymap(lambda x: [mapped_num[item] for item in x]).values.tolist())

print(train[0], '\n\nShape of the input:\n', train.shape)
train=np.transpose(train,(0, 2, 1))

print(train[0], '\n\nNew shape of the input:\n', train.shape)
inp = tf.keras.layers.Input(shape=(107,3))

embedding = tf.keras.layers.Embedding(14,100)(inp)

print(embedding)
inpfinal = tf.keras.backend.reshape(embedding,(-1,107,300))

print(inpfinal)
##Define root mean square error

def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return keras.backend.sqrt(mse)



##Define input

inp = tf.keras.layers.Input(shape=(107,3))

embedding = tf.keras.layers.Embedding(14,100)(inp)

inpfinal = tf.keras.backend.reshape(embedding,(-1,107,300))



##Define model

hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=.3,return_sequences=True))(inpfinal)

hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=.3,return_sequences=True))(hidden)

hidden = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,dropout=.3,return_sequences=True))(hidden)



##Output (ignore the markers...only the first 68 are scored)

out = tf.keras.layers.Dense(3, activation='linear')(hidden[:, :68])



model = tf.keras.Model(inputs=inp, outputs=out)

model.compile(tf.optimizers.Adam(), loss=rmse)

print(model.summary())