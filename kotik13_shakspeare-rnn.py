import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
file = open('../input/input.txt', 'r')

text = file.readlines()

text = [x.rstrip('\n') for x in text]

text = ' '.join(text)

text = text.lower()

alphabet = list(sorted(set(text)))

alphabet_size = len(alphabet)
type(text)
#for using embedding layer

model_encoder = LabelEncoder()

model_encoder.fit(np.asarray(alphabet).reshape(-1,1))

text_numerical = model_encoder.transform(np.asarray([ch for ch in text]).reshape(-1,1))
#without embedding layer, lstm only

#model_encoder = OneHotEncoder(handle_unknown='ignore')

#model_encoder.fit(np.asarray(alphabet).reshape(-1,1))

#text_numerical = model_encoder.transform(np.asarray([ch for ch in text]).reshape(-1,1)).toarray()
text_numerical

np.shape(text_numerical)
def chunks(l, n, truncate=False):

    batches = []

    for i in range(0, len(l), n):

        if truncate and len(l[i:i + n]) < n:

            continue

        batches.append(l[i:i + n])

    return batches
seqlen = 50

nbatch = 50

Xs = chunks(text_numerical, seqlen)

Ys = chunks(text_numerical[1:-1], seqlen)
Xs = np.stack(Xs[:-1], axis=0)

Ys = np.stack(Ys[:-1], axis=0)

#Xs = np.matrix.transpose(Xs)

#Ys = np.matrix.transpose(Ys)



#Xs = np.concatenate(Xs[:-1], axis=0)

#Ys = np.concatenate(Ys[:-1], axis=0)

np.shape(Xs), np.shape(Ys)
Xs.shape
import keras

import keras.backend as K

from keras.layers.recurrent import LSTM

from keras.layers.core import Dense,Flatten, Reshape

from keras.layers import SpatialDropout1D,Embedding,Dropout

from keras.models import Sequential

from keras.layers import TimeDistributed

from keras import regularizers
#without embedding layer, lstm only

m = Sequential()

m.add(LSTM(128, input_dim=alphabet_size, input_length=seqlen, return_sequences=True))

m.add(LSTM(128, input_dim=128, input_length=seqlen, return_sequences=True))

m.add(TimeDistributed(Dense(alphabet_size, activation='softmax')))

m.add(SpatialDropout1D(0.15))

m.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(lr=0.005))

m.fit(Xs,Ys, batch_size=nbatch, epochs=3)
#using Embedding layer

m = Sequential()

m.add(Embedding(input_dim=alphabet_size, output_dim=128, 

                embeddings_regularizer=regularizers.l2(0.001)))

m.add(LSTM(128, input_dim=128,return_sequences=False))

m.add((Dense(alphabet_size, input_dim=128, activation='softmax')))

m.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(lr=0.005))

m.fit(Xs,Ys,  epochs=1)
pr = m.predict(Xs[0:2,0:40])
np.shape(pr)
pr
for i in pr:

    print(np.random.choice(alphabet,p=i))
Xs[0:50]
m.summary()
phrase='Dursley were proud to say that they were perfectly normal, thack you very much.'[:40].lower()

el = model_encoder.transform(np.asarray([ch for ch in phrase]).reshape(-1,1))

#el = Xs[0,]
np.shape(el)
#el = el[np.newaxis, ...]

pred = m.predict(el,batch_size=seqlen)
np.shape(pred)
pred
#el = el[np.newaxis, ...]

final_str = ''

for i in range(len(phrase)):

    final_str=final_str+alphabet[np.argmax(el[-1,i])]  

for i in range(500):

    pred = m.predict(el,batch_size=seqlen)

    final_str=final_str + np.random.choice(alphabet,p=pred[-1,-1])

    new_el = np.concatenate((el, pred[:,-1,:][np.newaxis, ...]), axis=1)

    new_el = new_el[:,1:,:]

    el=new_el

final_str
m.summary()
el = el[np.newaxis, ...]
phrase='Dursley were proud to say that they were perfectly normal, thack you very much.'[:1].lower()

el = model_encoder.transform(np.asarray([ch for ch in phrase]).reshape(-1,1)).toarray()

el = el[np.newaxis, ...]

p = m.predict_on_batch(el)
text
pred[-1,-1]
4+5