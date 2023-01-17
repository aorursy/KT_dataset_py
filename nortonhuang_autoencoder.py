# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

os.listdir('../input/user-feature')

# Any results you write to the current directory are saved as output.
from time import time

from pandas import merge

import numpy as np

import keras.backend as K

from keras.engine.topology import Layer, InputSpec

from keras.layers import Dense, Input,Activation

from keras.models import Model,Sequential

from keras.optimizers import Adam

from keras.initializers import VarianceScaling

from keras import callbacks

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.cluster import KMeans



import os

import pandas as pd

import json
#build a autoencoder

def autoencoder(dims, act='relu', init='glorot_uniform'):

    """

    Fully connected auto-encoder model, symmetric.

    Arguments:

        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.

            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1

        act: activation, not applied to Input, Hidden and Output layers

    return:

        (ae_model, encoder_model), Model of autoencoder and model of encoder

    """

    n_stacks = len(dims) - 1

    # input

    input_img = Input(shape=(dims[0],), name='input')

    x = input_img

    # internal layers in encoder

    for i in range(n_stacks-1):

        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)



    # hidden layer

    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  

    # hidden layer, features are extracted from here



    x = encoded

    # internal layers in decoder

    for i in range(n_stacks-1, 0, -1):

        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        

    # output

    x = Dense(dims[0], kernel_initializer=init, activation='softmax', name='decoder_0')(x)

    decoded = x

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')

#load data

data_train_user = pd.read_csv("../input/user-feature/user_feature.csv")

data_train_user.head()

enc = OneHotEncoder(sparse=False,handle_unknown='ignore')

gender = pd.DataFrame(enc.fit_transform(data_train_user["gender"].values.reshape(-1,1)))

gender.astype(int).head()

conab = pd.DataFrame(enc.fit_transform(data_train_user["consuptionAbility"].values.reshape(-1,1)))

conab.astype(int).head()

conecttype = pd.DataFrame(enc.fit_transform(data_train_user["connectionType"].values.reshape(-1,1)))

conecttype.astype(int).head()

data_train_user = merge(data_train_user,gender,how='left',left_index=True,right_index=True)

data_train_user = merge(data_train_user,conab,how='left',left_index=True,right_index=True)

data_train_user = merge(data_train_user,conecttype,how='left',left_index=True,right_index=True)

data_train_user.drop(['gender',"consuptionAbility","connectionType"],axis=1,inplace=True)

data_train_user.head()



train=data_train_user.iloc[:,1:51]

train.head()
#create autoencoder parameters

dims = [train.shape[-1],500,500,2000,10]

init = VarianceScaling(scale=1. / 3., mode='fan_in',distribution='uniform')

pretrain_optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)

pretrain_epochs=100

batch_size=64

save_dir='./results'
#create auto-encoder

autoencoder, encoder = autoencoder(dims,init=init)

autoencoder.summary()



#model = model_5layers = Sequential()

#model.add(autoencoder)

#model.add(Dense(dims[1],kernel_initializer='he_normal'))

#model.add(Activation('sigmoid'))

#model.summary()



autoencoder.compile(optimizer=pretrain_optimizer, loss='categorical_crossentropy',metrics=['categorical_accuracy'])

autoencoder.fit(train,train,batch_size=batch_size,

                epochs=pretrain_epochs,validation_split=0.01)

autoencoder.save_weights('./ae_weights.h5')
#get the values

predict = encoder.predict(train)

predict = pd.DataFrame(predict)

predict['user_ID'] = data_train_user['userId']

cols = list(predict)

cols.insert(0,cols.pop(cols.index('user_ID')))

predict = predict.loc[:,cols]

predict.head()

#save the 10-demensional feature

predict.to_csv("user_feature_10dim.csv",index = False)