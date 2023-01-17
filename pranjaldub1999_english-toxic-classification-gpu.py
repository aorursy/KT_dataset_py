# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from __future__ import print_function

from builtins import range

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import transformers

from transformers import TFAutoModel, AutoTokenizer

#try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

#    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#    print('Running on TPU ', tpu.master())

#except ValueError:

 #   tpu = None



#if tpu:

 #   tf.config.experimental_connect_to_cluster(tpu)

  #  tf.tpu.experimental.initialize_tpu_system(tpu)

   # strategy = tf.distribute.experimental.TPUStrategy(tpu)

#else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

 #   strategy = tf.distribute.get_strategy()



#print("REPLICAS: ", strategy.num_replicas_in_sync)



#AUTO = tf.data.experimental.AUTOTUNE



# Data access

MODEL = 'PD_TOXIC_X_CLASS'

# Set your own project id here

#PROJECT_ID = 'pd@toxic1234'

#from google.cloud import storage

#storage_client = storage.Client(project=PROJECT_ID)

#from google.cloud import bigquery

#bigquery_client = bigquery.Client(project=PROJECT_ID)
import os,sys,numpy as np,pandas as pd

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import LSTM,Bidirectional,Dense,GlobalMaxPooling1D,Embedding,Input,MaxPooling1D

from keras.models import Model

from keras.optimizers import RMSprop

from sklearn.metrics import roc_auc_score

from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
#train data and test labels 

train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

train.head()

#load word vectors

word2vec = {}

with open(os.path.join('/kaggle/input/glove-vector/glove.6B.100d.txt')) as f:

    for line in f:

        values = line.split()

        word = values[0]

        vec = np.asarray((values [1:]))

        word2vec[word] = vec

print('total vocab in glove',len(word2vec))

        

        

        

valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')



y_valid = valid.toxic.values
print(train.shape)
train.head()
train.tail()
max_seq_len = 20

max_voc_size = 400000

embed_dim = 100

val_split = 0.2

batch_size = 128

epoch = 5
#extract sentences

sentences = train['comment_text'].fillna('DUMMY_VALUES').values

#target

labels = ['toxic' ,'severe_toxic' ,'obscene' ,'threat' ,'insult' ,'identity_hate']

target  = train[labels].values

print(sentences[1:2])

print(target[1:2])



#SENTENCE FOR TEST

#sentences_test = test['comment_text'].fillna('DUMMY_VALUES').values



#SENTENCE FOR validate

#sentences_validate = valid['comment_text'].fillna('DUMMY_VALUES').values

tokenizer = Tokenizer(num_words = max_voc_size)

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

#sequences test

#sequences_test = tokenizer.texts_to_sequences(sentences_test)



#sequences validate

#sequences_validate = tokenizer.texts_to_sequences(sentences_validate)
print(len(sequences))
#word2idx

word2idx = tokenizer.word_index
#padding 

padded_seq = pad_sequences(sequences,maxlen=max_seq_len)



#padding test data

#padded_seq_test = pad_sequences(sequences_test , maxlen = max_seq_len)



#padding the validate data

#padded_seq_validate = pad_sequences(sequences_validate , maxlen = max_seq_len)
print(padded_seq.shape)
#embedding matrix

num_words = min(max_voc_size,len(word2idx)+1)

embed_matrix = np.zeros((num_words,embed_dim))

for word , i in word2idx.items():

    if i < max_voc_size:

        #getting word vector

        word_vec = word2vec.get(word)

        if word_vec is not None:

            embed_matrix[i]=word_vec
#embedding layer

embed_layer = Embedding(num_words,embed_dim,

                        weights = [embed_matrix],

                       input_length=max_seq_len,

                       trainable = True)

#building model

from keras.optimizers import Adam

from keras.layers import Activation

import tensorflow as tf

# detect and init the TPU

#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#tf.config.experimental_connect_to_cluster(tpu)

#tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

#with tpu_strategy.scope():

    

input_ = Input(shape = (20,))

# = embed_layer(input_)

x = Embedding(num_words , 20,input_length=max_seq_len)(input_)

x = LSTM(20,return_sequences=True)(x)

x = MaxPooling1D(3)(x)

x = Bidirectional(LSTM(20,return_sequences = True))(x)

x = GlobalMaxPooling1D()(x)

x = Dense(6)(x)

out = Activation('sigmoid')(x)

model = Model(input_,out)

model.compile(loss = 'binary_crossentropy',

             optimizer = Adam(lr = 0.001),

             metrics = ['categorical_accuracy'])

model.summary()
#implementing callbacks

from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint = ModelCheckpoint("toxic_rnn.h5",

                            monitor="val_loss",

                            mode="min",

                            save_best_only=True,

                            verbose=1)

early_stopping = EarlyStopping(monitor="val_loss",

                              min_delta=0,

                              patience=3,

                              verbose=1,

                              restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",

                             factor=0.1,

                             patience=3,

                             verbose=1,

                             min_delta=0.0001)

#putting callbacks in callbacks list

callbacks = [checkpoint,early_stopping,reduce_lr]
r = model.fit(padded_seq,target,batch_size=batch_size,epochs=epoch,validation_split = 0.2 ,callbacks=callbacks)
model.save('/kaggle/working/toxic_softmax3_notglove.h5')
!pip install simplejson
import simplejson as json
tokenizer_json = tokenizer.to_json()

with open('tokenizer.json', 'w', encoding='utf-8') as f:

    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
from keras.models import load_model
model2 = load_model('/kaggle/working/toxic_softmax3_notglove.h5')
model2.summary()
sentences_new = train['comment_text'].fillna('DUMMY_VALUES').values
from keras.preprocessing.text import tokenizer_from_json

with open('tokenizer.json') as f:

    data = json.load(f)

    tokenizer4 = tokenizer_from_json(data)









#tokenizer2 = Tokenizer(num_words = 20000)

#tokenizer2.fit_on_texts(sentences_new)

sequences_cus = tokenizer4.texts_to_sequences(sentences_new)
word2idx = tokenizer2.word_index

print(word2idx[FUCKED])
#s = ['COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK']

s = ['I amm happy for you brother']

s2 = ['i will kill you motherfucker']
#tokenizer2 = Tokenizer(num_words =  20000)

#tokenizer.fit_on_texts(s)

sequences_custom = tokenizer4.texts_to_sequences(s)

sequences_custom2 = tokenizer4.texts_to_sequences(s2)

print(sequences_custom)

print(sequences_custom2)
padded_seq_cus = pad_sequences(sequences_custom,maxlen=20,padding  = 'post')

padded_seq_cus_2 = pad_sequences(sequences_custom2,maxlen=20,padding  = 'post')
print(padded_seq_cus_2)

print(padded_seq_cus_2.shape)

print(padded_seq_cus)

print(padded_seq_cus.shape)
x = model2.predict(padded_seq_cus)

x2 = model2.predict(padded_seq_cus_2)
print(x.shape)

print(x2.shape)
print(x)

print(x2)
def category(arr):

    label = ['toxic','severe','obscene','threat','insult','identity_hate']

    for a in arr:

        for x in range(6):

            print ('sentence is {} percent toxic of category {} '.format(a[x],label[x]))

print(category(x))

print(category(x2))
def loaded(name):

    from keras.preprocessing.text import tokenizer_from_json

    with open(name) as f:

        data = json.load(f)

        tokenizer_loaded = tokenizer_from_json(data)

        return tokenizer_loaded

def load_saved_model(location):

     model_infer = load_model('/kaggle/working/toxic_softmax3_notglove.h5')

     return model_infer



def pipeline(lis):

    name = 'tokenizer.json'

    location = '/kaggle/working/toxic_softmax3_notglove.h5'

    model_infer = load_saved_model(location)

    tokenizer_pipe = loaded(name)

    sequences_custom_pipe = tokenizer_pipe.texts_to_sequences(lis)

    padded_seq_cus_pipe = pad_sequences(sequences_custom_pipe,maxlen=20,padding  = 'post')

    pred = model_infer.predict(padded_seq_cus_pipe)

    category(pred)
lis = ['i want to kill you till death']

lis2 = ['motherfucker fuck you man']

lis3 = ['i am happy that you got the job']

pipeline(lis)

pipeline(lis2)

pipeline(lis3)