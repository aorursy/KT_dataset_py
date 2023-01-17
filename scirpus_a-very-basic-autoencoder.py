import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

train.drop(unwanted,inplace=True,axis=1)

test.drop(unwanted,inplace=True,axis=1)
highcardinality =[]

for i in train.columns[1:-1]:

    if(((i.find('bin')!=-1) or (i.find('cat')!=-1))):

        ln = len(pd.concat([train[i],test[i]]).unique())

        if((ln < 5)):

            highcardinality.append(i)

            print(i,len(train[i].unique()))

highcardinality
allcats = pd.concat([train[highcardinality],test[highcardinality]])
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

allcats[highcardinality] = ss.fit_transform(allcats[highcardinality])

allcats.shape
import tensorflow as tf

clusterencoder = None

input_dim = allcats.shape[1]

input_layer = Input(shape=(input_dim, ))

encoder = Dense(input_dim, activation="relu")(input_layer)

encoder = Dense((36), activation="relu")(encoder)

encoder = Dense((36), activation="relu")(encoder)

encoder = Dense((18), activation="relu")(encoder)

encoder = Dense((18), activation="relu")(encoder)

encoder = Dense((8), activation="relu")(encoder)

encoder = Dense(2, activation="tanh")(encoder)

clusterencoder = Model(inputs=input_layer, outputs=encoder)

decoder = Dense((8), activation='relu')(encoder)

decoder = Dense((18), activation='relu')(decoder)

decoder = Dense((18), activation='relu')(decoder)

decoder = Dense((36), activation='relu')(decoder)

decoder = Dense((36), activation='relu')(decoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 10

batch_size = 128

autoencoder.compile(optimizer='adam', 

                    loss='mean_squared_error', 

                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",

                               verbose=0,

                               save_best_only=True)

tensorboard = TensorBoard(log_dir='./logs',

                          histogram_freq=0,

                          write_graph=True,

                          write_images=True)

history = autoencoder.fit(allcats.values, allcats.values,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=True,

                    validation_data=(allcats[::100].values, allcats[::100].values),

                    verbose=2,

                    callbacks=[checkpointer, tensorboard]).history
a = clusterencoder.predict(allcats.values)

print(a.shape)

atrain = a[:train.shape[0]]

atest = a[train.shape[0]:]
targets = train.target.ravel()
colors = ['red','blue']
plt.figure(figsize=(15,15))

plt.scatter(atrain[:,0],atrain[:,1],color=[colors[x] for x in targets], alpha=.5)