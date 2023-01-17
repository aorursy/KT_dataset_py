# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import csv

import sys

import cv2

import os

import random

import re

import nltk

import pickle

import numpy as np

import pandas as pd

import tensorflow as tf

from numpy import array

from numpy import asarray

from numpy import zeros

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

from keras import backend as keras

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D,Embedding,LSTM

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.layers import GlobalMaxPooling1D

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from tensorflow.keras.layers import Embedding



tf.test.gpu_device_name()
#!conda install -y gdown
#import gdown

#url = 'https://drive.google.com/uc?export=download&id=1-NxLIxP1FZm9T-eC9u22ZvOqbTSnYTSi'

#output = 'model_w.h5'

#gdown.download(url, output, quiet=False)
!pip install '/kaggle/input/transformerswhlfiles/transformers-2.11.0-py3-none-any.whl'

def preprocess_train(train):

  #preprocessing for the train dataset

  train['text'] = train['text'].fillna('')

  train['selected_text'] = train['selected_text'].fillna('')

  train['sentiment']=train['sentiment'].replace('neutral',0)

  train['sentiment']=train['sentiment'].replace('positive',1)

  train['sentiment']=train['sentiment'].replace('negative',2)

  #sns.countplot(x='sentiment', data=train)

  #plt.show()

  return train

def preprocess_test(test):

  #preprocessing for the train dataset

  test['text'] = test['text'].fillna('')

  test['sentiment']=test['sentiment'].replace('neutral',0)

  test['sentiment']=test['sentiment'].replace('positive',1)

  test['sentiment']=test['sentiment'].replace('negative',2)

  #sns.countplot(x='sentiment', data=test)

  #plt.show()

  return test
#analysing

train_original=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

train_original=preprocess_train(train_original)



test_original=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

test_original=preprocess_test(test_original)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../input/saved-pretrains')

vocabulary = tokenizer.get_vocab()

print(list(vocabulary.keys())[5000:5020])
max_length = 35

batch_size = 6
def convert_example_to_feature(review):

  # combine step for tokenization, WordPiece vector mapping, adding special tokens as well as truncating reviews longer than the max length

  return tokenizer.encode_plus(review, 

                add_special_tokens = True, # add [CLS], [SEP]

                max_length = max_length, # max length of the text that can go to BERT

                pad_to_max_length = True, # add [PAD] tokens

                return_attention_mask = True, # add attention mask to not focus on pad tokens

              )
def encode_examples_train(ds):

  input_ids_list = []

  token_type_ids_list = []

  attention_mask_list = []

  label_list = []

  all_list=[]

  for review,selected,sent in (ds[['text','selected_text','sentiment']]).itertuples(index=False):

    bert_input = convert_example_to_feature(review)

    input_ids_list=(bert_input['input_ids'])

    token_type_ids_list=(bert_input['token_type_ids'])

    attention_mask_list=(bert_input['attention_mask'])

    bertselect=convert_example_to_feature(selected )

    label_list.append([sent*x for x in bertselect['attention_mask']])

    all_list.append([input_ids_list,token_type_ids_list,attention_mask_list])

  return all_list,label_list

def encode_examples_test(ds):

  input_ids_list = []

  token_type_ids_list = []

  attention_mask_list = []

  all_list=[]

  for review,sent in (ds[['text','sentiment']]).itertuples(index=False):

    bert_input = convert_example_to_feature(review)

    input_ids_list=(bert_input['input_ids'])

    token_type_ids_list=(bert_input['token_type_ids'])

    attention_mask_list=(bert_input['attention_mask'])

    all_list.append([input_ids_list,token_type_ids_list,attention_mask_list])

  return all_list



# train dataset

ds_train_encoded,labels_train = encode_examples_train(train_original)



# test dataset

ds_test_encoded = encode_examples_test(test_original)
def jaccard_distance(y_true, y_pred, smooth=100):

    """ Calculates mean of Jaccard distance as a loss function """

    intersection = tf.reduce_sum(y_true * y_pred, axis=(1))

    sum_ = tf.reduce_sum(y_true + y_pred, axis=(1))

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    jd =  (1 - jac) * smooth

    return tf.reduce_mean(jd)
def jaccard_score(y_true, y_pred, smooth=100):

    """ Calculates mean of Jaccard distance as a loss function """

    arr1=np.array([np.array(xi) for xi in y_true])

    arr1=(arr1>0).astype('int')

    arr2=np.array([np.array(xi) for xi in y_pred])

    arr2=(arr2>0).astype('int')

    intersection = np.sum(np.multiply(arr1 , arr2), axis=(1))

    sum_ =np.sum(arr1,axis=1)+np.sum(arr2,axis=1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth) #score

    jd =  (1 - jac) * smooth #distance

    return (jd)
from transformers import TFBertModel,BertConfig

import tensorflow as tf



from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dense,Conv2D,Reshape,MaxPooling2D,Flatten,Conv1D,MaxPooling1D,Dropout

learning_rate = 2e-5



all_ins = Input(shape = (3,max_length,), dtype=tf.int32)

ids = all_ins[:,0,:]

att =  all_ins[:,1,:]

tok =  all_ins[:,2,:]



'''

    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

'''

#config = BertConfig.from_pretrained('kaggle/input/saved-pretrains/config.json')

config = BertConfig() # print(config) to see settings

config.output_hidden_states = False # 

bert = TFBertModel.from_pretrained('/kaggle/input/bert-base-uncased/bert-base-uncased-tf_model.h5',config=config)([ids,att,tok])



#bert = TFBertModel.from_pretrained('bert-base-uncased')([ids,att,tok]) 

bert = bert[0]



conv1=Conv1D(512,(3),padding='same',activation='relu')(bert)

drop1=Dropout(0.5)(conv1)

conv2=Conv1D(256,(3),padding='same',activation='relu')(drop1)

drop2=Dropout(0.5)(conv2)

conv3=Conv1D(64,(3),padding='same',activation='relu')(drop2)

classifier = Dense(3, activation='softmax')(conv3)



model = Model(all_ins, outputs=classifier)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[metric])

model.summary()



model.load_weights('/kaggle/input/bert-classifier/save_model_w.h5')

#yt=np.array(labels_train)

#y= yt.reshape(yt.shape[0], yt.shape[1], 1)

#y= yt.reshape(yt.shape[0], 1)

#bert_history = model.fit( ds_train_encoded,y,batch_size=4,epochs=6)

model.save_weights('/kaggle/working/save_model_w.h5')


#url = 'https://drive.google.com/uc?export=download&id=1GR-R-rr8gKx2luKRlDs5af4o1t-YVDIp'

#output = 'berty.h5'

#gdown.download(url, output, quiet=False)



Y_TEST=model.predict(ds_train_encoded[10:20])

pred=np.argmax(Y_TEST,axis=-1)

print(pred)

print(labels_train[0:10])

jac_des=jaccard_score(pred,labels_train[10:20])

print(jac_des)

print(np.mean(jac_des))
Y_TEST_blind=model.predict(ds_test_encoded)

Y_TEST_blind_pred=np.argmax(Y_TEST_blind,axis=-1)
#convert the output

output=[]

output_decoded=[]

sent_list=[]

i=0

for item in ds_test_encoded:

  j=0

  temparr=[]

  sent=0

  for item2 in item[0]:

    if Y_TEST_blind_pred[i][j]>0:

      if (item2!=102) and (item2!=101): 

        temparr.append(item2)

      sent=Y_TEST_blind_pred[i][j]

    j=j+1

  output.append(temparr.copy())

  if (sent!=0) and test_original['sentiment'][i]!=0 :

    output_decoded.append(tokenizer.decode(temparr))

  else:

    output_decoded.append(test_original['text'][i])

  sent_list.append(sent)

  i=i+1

z=0

print(test_original['text'][z])

print(ds_test_encoded[z][0])

print(Y_TEST_blind_pred[z])

print(output[z])

print(output_decoded[z])

print(sent_list[z])
from pandas import DataFrame

df = DataFrame({'textID': test_original['textID'], 'selected_text': output_decoded})

df.to_csv('submission.csv', index=False)