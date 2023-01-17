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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from gensim.models import Word2Vec

from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.test.utils import common_texts, get_tmpfile

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn import metrics

from keras.layers import *

from keras.models import *

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import *

from keras.optimizers import *

import keras.backend as K

from keras.callbacks import *

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from smart_open import smart_open

import datetime 

from keras.utils import multi_gpu_model



import os

import time

import gc

import re

import random

#from unidecode import unidecode
def convertCats2sentence(x,cols):

    

    ref = [[col.lower() + '=' + (str(val)).lower() for val in x[col]] for col in cols]



    # reformat list to be sentence lines, docs for doc2vec

    

    tmp = pd.DataFrame.from_records(ref)

    

    tmp = tmp.transpose()

    

    X = tmp.values.tolist()

    return X





#filename = '../input/train.csv'



#object_key = 'train.csv'



#path = 's3://{}:{}@{}/{}'.format(aws_key, aws_secret, bucket_name, object_key)



chunksize = 100000

header = pd.read_csv('../input/microsoft-malware-prediction/train.csv',nrows=1)

cols = list(set(header.head(0)) - set(['Census_PrimaryDiskTotalCapacity',

            'Census_SystemVolumeTotalCapacity','HasDetections']))



n =  8921484 - 1 

s = 500000 #desired sample size # low just for show...

skip = sorted(random.sample(range(1,n+1),n-s)) 



X = []

t0 = time.time()

steps = 1

#smart_open(path)

for chunk in pd.read_csv('../input/microsoft-malware-prediction/train.csv', chunksize=chunksize,dtype='category',skiprows=skip):

    # chunk is a dataframe

    print('step...'+str(steps))

    X.extend(convertCats2sentence(chunk,cols))

    steps = steps + 1

    

t1 = time.time()

print(t1-t0)

print('train converted...')

gc.collect()
embed_size = 300

maxlen = len(cols)

max_features = None

window_size = 40

t0 = time.time()

model = Word2Vec(X,size=embed_size,window=window_size,min_count=1)

t1 = time.time()

print(t1-t0)

print('model fit...')

## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features, filters='')

tokenizer.fit_on_texts(X)



# read in all training data...for model training



n =  8921484 - 1 

s = 1000000 #desired sample size # low just for show

skip = sorted(random.sample(range(1,n+1),n-s)) 



X = []

t0 = time.time()

steps = 1

#smart_open(path)

gc.collect()

for chunk in pd.read_csv('../input/microsoft-malware-prediction/train.csv', chunksize=chunksize,dtype='category',skiprows=skip):

    # chunk is a dataframe

    print('step...'+str(steps))

    X.extend(convertCats2sentence(chunk,cols))

    steps = steps + 1

    

t1 = time.time()

print(t1-t0)

print('train converted...')





X = tokenizer.texts_to_sequences(X)

X = pad_sequences(X, maxlen=maxlen)



word_index = tokenizer.word_index

max_features = len(word_index)+1



train = pd.read_csv('../input/microsoft-malware-prediction/train.csv',usecols=['HasDetections'],skiprows=skip)

Y = train['HasDetections']

del train

gc.collect()
def load_wv(model, word_index):

    embedding_matrix = np.random.normal(model.wv[model.wv.vocab].mean(), model.wv[model.wv.vocab].std(), 

                                        (max_features, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

            

        try: 

            embedding_vector = model.wv[word]

            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        except:

            print(word, ' not found')

            

            

    return embedding_matrix 



embedding_matrix = load_wv(model, word_index)
embedding_matrix.shape
#object_key = 'test.csv'



#path = 's3://{}:{}@{}/{}'.format(aws_key, aws_secret, bucket_name, object_key)



#t0 = time.time()

#X_test = []

#for chunk in pd.read_csv('test.csv', chunksize=chunksize,dtype='category'):

    # chunk is a dataframe

 #   X_test.extend(convertCats2sentence(chunk,cols))

    

#t1 = time.time()

#print(t1-t0)

#print('test converted...')



#X_test = tokenizer.texts_to_sequences(X_test)

#X_test = pad_sequences(X_test, maxlen=maxlen)



sub = pd.read_csv('../input/microsoft-malware-prediction/test.csv', usecols=['MachineIdentifier'])



#gc.collect()
def squash(x, axis=-1):

    # s_squared_norm is really small

    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)

    # return scale * x

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



# A Capsule Implement with Pure Keras

class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, 

                 kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)







def capsule():

    K.clear_session()       

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, 

                  weights=[embedding_matrix], trainable=False)(inp)

    x = SpatialDropout1D(rate=0.2)(x)

    x = Bidirectional(CuDNNLSTM(100, return_sequences=True, 

                                kernel_initializer=glorot_normal(seed=12300), 

                                recurrent_initializer=orthogonal(gain=1.0, 

                                                                 seed=10000)))(x)



    x = Capsule(num_capsule=10, dim_capsule=5, routings=2, share_weights=True)(x)

    x = Flatten()(x)



    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)

    x = Dropout(0.12)(x)

    x = BatchNormalization()(x)

    

    x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)

    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)



    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=Adam(),)

    

    return model



def auc(y_true, y_pred):

    

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)



    return metrics.auc(fpr, tpr)
kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

bestscore = []

y_test = np.zeros((sub.shape[0], ))

for i, (train_index, valid_index) in enumerate(kfold.split(X, Y)):

    X_train, X_val, Y_train, Y_val = X[train_index], X[valid_index], Y[train_index], Y[valid_index]

    filepath="weights_best.h5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', 

                                 verbose=2, save_best_only=True, mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=1, 

                                  min_lr=0.0001, verbose=2)

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, 

                                  patience=2, verbose=2, mode='auto')

    callbacks = [checkpoint, reduce_lr]

    with tf.device('/cpu:0'):

        model = capsule()

    if i == 0:print(model.summary()) 

    

    

    #parallel_model = multi_gpu_model(model, gpus=2)

    #parallel_model.compile(loss='binary_crossentropy', optimizer=Adam())

    #parallel_model.fit(X_train, Y_train, batch_size=512, epochs=6, 

    #                   validation_data=(X_val, Y_val), verbose=2, callbacks=callbacks )

    #parallel_model.load_weights(filepath)

    model.fit(X_train, Y_train, batch_size=512, epochs=20, 

                       validation_data=(X_val, Y_val), verbose=2, callbacks=callbacks )

    y_pred = model.predict([X_val], batch_size=1024, verbose=2)

    #sub['HasDetections']  = np.squeeze(model.predict([X_test], batch_size=1024, verbose=2))

    

   # predict in batches to avoid memory overhead

    

    t0 = time.time()

    yhat = []

    step = 1

    for X_test in pd.read_csv('../input/microsoft-malware-prediction/test.csv', chunksize=chunksize,dtype='category'):

        X_test = pad_sequences(tokenizer.texts_to_sequences(

            convertCats2sentence(X_test,cols)),maxlen=maxlen)

        

        yhat.extend(np.squeeze(model.predict([X_test],batch_size=1024,verbose=2)))

        #print('predicted partition ',step,'...')

        #step = step + 1

        

    

    t1 = time.time()

    print('test predicted....',t1-t0)



    y_test += np.array(yhat)/5

    auc_val = auc(np.squeeze(Y_val), np.squeeze(y_pred))

    

    print('AUC: {:.4f}'.format(auc_val))

    bestscore.append(auc_val)

    gc.collect() # this is necessary to prevent crashing when using full dataset

    

    #now = datetime.datetime.now()

    

    #object_key = 'submission_' + str(i) + '_' + str(now) + '.csv'



    #path = 's3://{}:{}@{}/{}'.format(aws_key, aws_secret, bucket_name, object_key)



    #sub.to_csv(smart_open(path,mode='w'), index=False)

    
sub['HasDetections']  = y_test.reshape((-1, 1))

tmp = pd.read_csv('../input/lgbyhat/lgb_submission.csv')

tmp.columns = ['MachineIdentifier','HD']

sub = pd.merge(sub,tmp)

sub['mean'] = sub.mean(axis=1)

sub.drop(['HD','HasDetections'],axis=1,inplace=True)

sub.columns = ['MachineIdentifier','HasDetections']

sub.to_csv("submission.csv", index=False)