from keras.layers import Dense ,LSTM,concatenate,Input,Flatten

import tensorflow as tf

import matplotlib

import numpy as np

import pandas as pd

from keras.models import Model

from keras.callbacks import EarlyStopping

#

def model(i,p,data_a,data_b,labels):

    #x=tf.placeholder(tf.float32,shape=(i,1,None))

    x=Input(shape=(i,1))

    #x = Flatten()(x)

    y=Input(shape=(i,1))

    #y = Flatten()(y)

    #y=tf.placeholder(tf.float32,shape=(i,1,None))

    #LSTM layers

    admi=LSTM(40,return_sequences=False)(x)

    pla=LSTM(40,return_sequences=False)(y)

    out=concatenate([admi,pla],axis=-1)

    print(pla)

    #out=Flatten(input_shape=(None,1))(out)

    print(out)

    #out=np.reshape(out,(1,9))

    output=Dense(1, activation='sigmoid')(out)

    model = Model(inputs=[x, y], outputs=output)

    model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

    n=model.fit([data_a, data_b],labels,batch_size=1, epochs=10)

    return n

data_a=np.array([300,455,350,560,700,800,200,250,300])

labes=np.array([455,350,560,700,800,200,250,300,350])

data_b=np.array([200,255,350,470,600,300,344,322,300])

data_a=np.reshape(data_a,(9,1,1))

data_b=np.reshape(data_b,(9,1,1))

#s=labes.shape()

#labes=np.reshape(labes,(1,9,1))

print(data_a.shape)

print(data_b.shape)

print(labes.shape)



#labes=labes(9,4)

#data_b=tf.placeholder(tf.float32,shape=(6,1))



#labes=tf.placeholder(tf.float32,shape=(6,1))



re=model(1,4,data_a,data_b,labes)

print(re)