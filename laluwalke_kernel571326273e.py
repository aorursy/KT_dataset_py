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
## Import all libraries

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.layers import Dense,Input,Embedding

from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

import keras.backend as K

from transformers import *

import tokenizers

from tokenizers import ByteLevelBPETokenizer

import os

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder
## Importing the dataset.

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv', keep_default_na=False)

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv', keep_default_na=False)
#Config

Max_length = 120
##All functions 

## function returning overlap between two id vectors

def list_in(a, b):

  y_seq = np.zeros((len(b)),dtype='int32')

  for i in range(0,len(b) - len(a) + 1):

    if b[i:i + len(a)] == a:

      y_seq[i:i+len(a)] = 1

      if b[i-1] == " ":

        y_seq[i-1] = 1

      break

  return y_seq

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
### Making Dataset for training on complete dataset(preprocessing for roberta)

## Defining the tokenizer.

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file = "/kaggle/input/roberta-base/vocab-roberta-base.json" ,

    merges_file = "/kaggle/input/roberta-base/merges-roberta-base.txt",

    lowercase=True,

    add_prefix_space=True)
Max_length = 120

obs = train.shape[0]

input_tokens = np.ones((obs,Max_length),dtype='int32')

mask = np.zeros((obs,Max_length),dtype='int32')

seg = np.zeros((obs,Max_length),dtype='int32')

st_vec = np.zeros((obs,Max_length),dtype='int32')

end_vec = np.zeros((obs,Max_length),dtype='int32') 



for i in range(0,obs):

  ## Overlap character-wise.

  overlap_char = list_in(" ".join(train.selected_text.values[i].split())," "+" ".join(train.text.values[i].split()))



  ## Encoding the tweet

  tweet_ex = tokenizer.encode(" "+" ".join(train.text.values[i].split()))



  ## Tweet ids 

  tweet_ex_ids = tweet_ex.ids

  input_tokens[i,0:len(tweet_ex_ids)+5] = [0]+tweet_ex_ids+[2,2]+tokenizer.encode(train.sentiment.values[i]).ids + [2]

  mask[i,0:len(tweet_ex_ids)+5] = 1

    

  ## Tweet Offsets

  tweet_ex_off = tweet_ex.offsets



  ## Encoding selected part of tweet 

  selected_text_tweet = tokenizer.encode(" "+" ".join(train.selected_text.values[i].split()))



  tokens = []

  for j,(a,b) in enumerate(tweet_ex_off):

    if sum( overlap_char[a:b] ) > 0:

      tokens.append(j)



  ## Start Vector and End Vector

  if len(tokens)>0:

    st_vec[i,tokens[0]+1] = 1

    end_vec[i,tokens[-1]+1] = 1
obs1 = test.shape[0]

input_tokens_test = np.ones((obs1,Max_length),dtype='int32')

mask_test = np.zeros((obs1,Max_length),dtype='int32')

seg_test = np.zeros((obs1,Max_length),dtype='int32')



for i in range(0,obs1):

  ## Encoding the tweet

  tweet_ex_test = tokenizer.encode(" "+" ".join(test.text.values[i].split()))



  ## Tweet ids 

  tweet_ex_ids_test = tweet_ex_test.ids

  input_tokens_test[i,0:len(tweet_ex_ids_test)+5] = [0]+tweet_ex_ids_test+[2,2]+tokenizer.encode(test.sentiment.values[i]).ids + [2]

  mask_test[i,0:len(tweet_ex_ids_test)+5] = 1
## Model Building(MODEL_1)(Sentiment used in tokens)

def Create_Model():

  token_inputs = Input((Max_length,), dtype=tf.int32, name='token_inputs')

  mask_inputs = Input((Max_length,), dtype=tf.int32, name='mask_inputs')

  seg_inputs = Input((Max_length,), dtype=tf.int32, name='seg_inputs')

  # sentiment_input = Input((Max_length,2,),dtype=tf.float32,name='sentiment_inputs')



  config = BertConfig.from_pretrained("/kaggle/input/roberta-base/config-roberta-base.json") 

  model = TFRobertaModel.from_pretrained("/kaggle/input/roberta-base/pretrained-roberta-base.h5",config=config )



  pool_output,seq_out = model([token_inputs, mask_inputs, seg_inputs])

  # x1 = tf.keras.layers.concatenate([pool_output,sentiment_input])

  x1 = tf.keras.layers.Dropout(0.1)(pool_output) 

  x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)

  x1 = tf.keras.layers.LeakyReLU()(x1)

  x1 = tf.keras.layers.Conv1D(64,2,padding='same')(x1)

  x1 = tf.keras.layers.Dense(1)(x1)

  x1 = tf.keras.layers.Flatten()(x1)

  x1 = tf.keras.layers.Activation('softmax')(x1)



  # x2 = tf.keras.layers.concatenate([pool_output,sentiment_input])

  x2 = tf.keras.layers.Dropout(0.1)(pool_output) 

  x2 = tf.keras.layers.Conv1D(128, 2, padding='same',activation='relu')(x2)

  x2 = tf.keras.layers.LeakyReLU(0.1)(x2) 

  x2 = tf.keras.layers.Conv1D(64,2,padding='same',activation='relu')(x2)

  x2 = tf.keras.layers.Dense(1)(x2)

  x2 = tf.keras.layers.Flatten()(x2)

  x2 = tf.keras.layers.Activation('softmax')(x2)



  bert_model1 = Model([token_inputs, mask_inputs, seg_inputs],[x1,x2])

  return bert_model1
def train_fn(tokens,mask,seg,start_vec,end_vec,sentiment,tokens_test,mask_test,seg_test,start_vec_test,end_vec_test,sentiment_test,fold): 

    # tokens_tr = []

    # mask_tr = []

    # seg_tr = []

    # start_vec_tr = []

    # end_vec_tr = []

    # sentiment_tr = []

    # for i in range(0,len(tokens)):

    #   #if sentiment.iloc[i] != "neutral":

    #   if sentiment[i] != "neutral":

    #     tokens_tr.append(tokens[i])

    #     mask_tr.append(mask[i]) 

    #     seg_tr.append(seg[i])

    #     start_vec_tr.append(start_vec[i])

    #     end_vec_tr.append(end_vec[i])

    # tokens_tr = np.asarray(tokens_tr, dtype=np.int32) 

    # mask_tr = np.asarray(mask_tr, dtype=np.int32) 

    # seg_tr = np.asarray(seg_tr, dtype=np.int32) 

    # start_vec_tr = np.asarray(start_vec_tr, dtype=np.int32) 

    # end_vec_tr = np.asarray(end_vec_tr, dtype=np.int32) 



    # tokens_tst = []

    # mask_tst = []

    # seg_tst = []

    # start_vec_tst = []

    # end_vec_tst = []

    # for i in range(0,len(tokens_test)):

    #   if sentiment_test[i] != "neutral":

    #     tokens_tst.append(tokens[i])

    #     mask_tst.append(mask[i]) 

    #     seg_tst.append(seg[i])

    #     start_vec_tst.append(start_vec[i])

    #     end_vec_tst.append(end_vec[i])

    # tokens_tst = np.asarray(tokens_tst, dtype=np.int32) 

    # mask_tst = np.asarray(mask_tst, dtype=np.int32) 

    # seg_tst = np.asarray(seg_tst, dtype=np.int32) 

    # start_vec_tst = np.asarray(start_vec_tst, dtype=np.int32) 

    # end_vec_tst = np.asarray(end_vec_tst, dtype=np.int32) 



    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)



    if fold > 0:

      tf.tpu.experimental.initialize_tpu_system(resolver)



    with strategy.scope():

      bert_model1 = Create_Model()

      #bert_model2.compile(optimizer=optimizer,loss = "binary_crossentropy")

      bert_model1.compile(optimizer=optimizer,loss = "binary_crossentropy")

    

    # bert_model1 = Create_Model()

    # #bert_model2.compile(optimizer=optimizer,loss = "binary_crossentropy")

    # bert_model1.compile(optimizer=optimizer,loss = "binary_crossentropy")



    def scheduler(epoch):

      return 3e-5 * 0.2**epoch

    DISPLAY = 1

    # K.clear_session()

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)



    # bert_model2.fit([tokens_tr,mask_tr,seg_tr,sentiment_tr],[start_vec_tr,end_vec_tr],epochs=5,batch_size =8,verbose=DISPLAY,callbacks=[reduce_lr,cp_callback])

  #For Roberta Model

    bert_model1.fit([tokens,mask,seg],[start_vec,end_vec],epochs=5,batch_size =32,verbose=DISPLAY,callbacks=[reduce_lr,cp_callback],

                      validation_data = ([tokens_test,mask_test,seg_test],[start_vec_test,end_vec_test]))

    return bert_model1
## When submitting on kaggle

def prediction_fn(tokens_test,mask_test,seg_test,sentiment,text):

  tokens_tst = []

  mask_tst = []

  seg_tst = []

  txt = []

  for i in range(0,len(tokens_test)):

    if sentiment[i] != "neutral":

      tokens_tst.append(tokens_test[i])

      mask_tst.append(mask_test[i]) 

      seg_tst.append(seg_test[i])

      txt.append(text[i])

  tokens_tst = np.asarray(tokens_tst, dtype=np.int32) 

  mask_tst = np.asarray(mask_tst, dtype=np.int32) 

  seg_tst = np.asarray(seg_tst, dtype=np.int32)  



  #prediction = bert_model2.predict([tokens_tst,mask_tst,seg_tst,sentiment_tr],verbose=1)

  prediction = bert_model1.predict([tokens_tst,mask_tst,seg_tst],verbose=1)



  prediction1 =[]

  for i in range(0,len(prediction[0])):

    a = np.argmax(prediction[0][i])

    b = np.argmax(prediction[1][i])

    if b>=a:

      #predict1 = tokenizer.decode(tokens_tst[i][a:b+1])

      text1 = " "+" ".join(txt[i].split())

      enc = tokenizer.encode(text1)

      predict1 = tokenizer.decode(enc.ids[a-1:b])

    else:

      predict1 = txt[i]

    prediction1.append(predict1)

  return {'prediction_string':prediction1,

          'prediction':prediction} 
def evaluation(prediction_string,test_text,sentiment,text):

  flag = 0

  total_prediction_string = []

  for i in range(0,len(test_text)):

    if sentiment[i] == 'neutral':

      total_prediction_string.append(text[i])

      # print(i,test_text[i])

    if sentiment[i] != 'neutral':

      total_prediction_string.append(prediction_string[flag])

      # print(i,prediction_string[flag])

      flag+=1

  jaccard_metric = []

  for i in range(0,len(total_prediction_string)):

    jaccard_metric.append(jaccard(total_prediction_string[i],test_text[i]))

  return sum(jaccard_metric)/len(jaccard_metric)
models_path = '/kaggle/input/roberta-model-v16/'

length = len(test[test.sentiment!='neutral'].sentiment)

pred0 = np.zeros((length,Max_length))

pred1 = np.zeros((length,Max_length))

for i in range(0,5):

  bert_model1 = Create_Model()

  print('loading model weights')

  bert_model1.load_weights(models_path + 'v16-roberta-%i.h5'%(i))

  print('making predictions')

  predictions = prediction_fn(input_tokens_test,mask_test,seg_test,test.sentiment.values,test.text.values)["prediction"]

  prediction_string = prediction_fn(input_tokens_test,mask_test,seg_test,test.sentiment.values,test.text.values)["prediction_string"]

  pred0 += predictions[0]/5

  pred1 += predictions[1]/5
prediction1 = []

## converting predictions to binary form.

flag = 0

for i in range(0,len(test.text)):

  if test.sentiment.values[i] != 'neutral':

    a = np.argmax(pred0[flag])

    b = np.argmax(pred1[flag])

    flag += 1

    if b>=a:

      predict1 = tokenizer.decode(input_tokens_test[i][a:b+1])

    else:

      predict1 = test.text[i]

    prediction1.append(predict1)

  if test.sentiment.values[i] == 'neutral':

    prediction1.append(test.text.iloc[i])
submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
submission.selected_text = prediction1
submission.to_csv("submission.csv",index=False)