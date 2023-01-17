# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

from tensorflow.keras import backend as k

from tensorflow.keras import *

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import *

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import matplotlib.pyplot as plt

import nltk

nltk.download('wordnet')

nltk.download('stopwords')

nltk.download('punkt')

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import re

from string import punctuation

from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split

# from emoji import *

# import emoji

import functools

import string

import operator

import random

random.seed(50)
def clean_text(text):

    text = text.lower() 

    text = re.sub(r"(@[a-z]*)", "<mention>", text)#remove any word start with @

    text = re.sub(r"(&[a-z;]*)", "<none>", text)#remove any word start with &

    text = re.sub(r"(#[a-z;]*)", "<hash>", text)#remove any word start with #

    text = re.sub(r"(http|https|ftp|ftps)\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?", "<link>", text)#remove LINKs

    text = re.sub(r'https?://\S+', '<link>', text) # remove https? links

    text = re.sub(r"(www.[a-z.\/0-9]*)", "<link>", text)#remove LINKs

    return text

def Transfrom_text(textT,textS,type):

    t = " ".join("<tok>" for i in range (len(textS.split())))

    textT = textT.replace(textS,t)

    return textT

def preprocess_text(df):

    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))

    df['selected_text_clean'] = df['text'].apply(lambda x: clean_text(x))

    df['target'] = pd.DataFrame([Transfrom_text(df['text'][i],df['selected_text'][i],df['sentiment'][i]) for i in range(len(df))])

    return df 
train_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

print(test_df.isnull().sum())



train_df = train_df[train_df.sentiment != 'neutral']

train_df = train_df.dropna()

train_df = train_df.reset_index()



# train_df.text = "<startsent>"+ " " + train_df.text+" "+"<endsent>"

# test_df.text = "<startsent>"+" " + test_df.text+" "+"<endsent>"



train = preprocess_text(train_df)

test = test_df

print(train_df.shape)

print(train.shape, test.shape)

print(train.isnull().sum())
def load_glove():

  tqdm.pandas()

  f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt')

#   f = open('/kaggle/input/glove6b300dtxt/glove.6B.300d.txt')

    

  embedding_values = {}

  for line in tqdm(f):

      value = line.split(' ')

      word = value[0]

      coef = np.array(value[1:],dtype = 'float32')

      embedding_values[word] = coef

  return embedding_values
def fit_glove(embedding_values):

  

  all_embs = np.stack(embedding_values.values())

  emb_mean,emb_std = all_embs.mean(), all_embs.std()

  emb_mean,emb_std

  

  embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, 300))

  OFV = []

  for word,i in tqdm(token.word_index.items()):

      values = embedding_values.get(re.sub(r"[^A-Za-z]", "", word))

      if values is not None:

          embedding_matrix[i] = values

      else:

        OFV.append(word)

  print(len(OFV))

#   print(" | ".join(i for i in OFV))

  return embedding_matrix
t1 = pd.DataFrame({'A': test.text.values})

t2 = pd.DataFrame({'A': train.text})

t3 = pd.DataFrame({'A': train.target})

# print(t2)

all_tokens = pd.concat([t1,t2,t3],axis = 0)
token = Tokenizer(num_words=54000,filters='')

token.fit_on_texts(all_tokens.A) 

vocab_size = len(token.word_index)+1

vocab_size
reverse_input_char_index = token.index_word

def decode_sequence(train_pad_seq_x,input_seq):

  decoded_sentence = ""

  for i in range(len(input_seq)):

    # print(input_seq[i])

    if (input_seq[i] == 1 or input_seq[i] == 2):

      if train_pad_seq_x[i] != 0:

        sampled_char = reverse_input_char_index[train_pad_seq_x[i]]

        decoded_sentence += sampled_char + " "

  return decoded_sentence
embedding_values = load_glove()

embedding_matrix = fit_glove(embedding_values)
MAX_LEN = 35

for word,i in token.word_index.items():

  if "<tok>" in word:

    token.word_index[word] = token.word_index["<tok>"]    

    

train_seq_x = token.texts_to_sequences(train.text)

train_pad_seq_x = pad_sequences(train_seq_x,maxlen=MAX_LEN)



train_seq_y = token.texts_to_sequences(train.target)

train_pad_seq_y = pad_sequences(train_seq_y,maxlen=MAX_LEN)





train_pad_seq_y[train_pad_seq_y != token.word_index['<tok>']] = 0

train_pad_seq_y[train_pad_seq_y == token.word_index['<tok>']] = 1  
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
def validation():

    test_pred = model.predict(X_train).round().astype(int)

    avg = 0

    for i in range(len(X_train)):  

        st1 = decode_sequence(X_train[i],test_pred[i])

        st2 = decode_sequence(X_train[i],y_train[i])

        avg += jaccard(st1,st2)

    print("Jac train sccore = " , np.sum(avg) / len(X_train))



    test_pred = model.predict(X_test).round().astype(int)

    avg = 0

    for i in range(len(X_test)):  

        st1 = decode_sequence(X_test[i],test_pred[i])

        st2 = decode_sequence(X_test[i],y_test[i])

        avg += jaccard(st1,st2)

    print("Jac valid sccore = ",np.sum(avg) / len(X_test))
HIDDEN_DIM=256

    

inputs = Input(shape=(MAX_LEN, ), dtype='int32')



embedding_layer = Embedding(vocab_size,300,weights = [embedding_matrix],trainable = False)

encoder_LSTM_1 = Bidirectional(LSTM(HIDDEN_DIM,return_sequences=True,kernel_regularizer=regularizers.l2(0.01)))



dense_layer_relu = TimeDistributed(Dense(64, activation='relu'))

dense_layer_relu_1 = TimeDistributed(Dense(64, activation='relu'))



Drop = TimeDistributed(Dropout(0.2))

dense_layer = TimeDistributed(Dense(1, activation='sigmoid'))



encoder_embedding = embedding_layer(inputs)



Encoded_seq = encoder_LSTM_1(encoder_embedding)



outputs = Drop(dense_layer_relu(Encoded_seq))

outputs = Drop(dense_layer_relu_1(outputs))



outputs = dense_layer(outputs)



model = Model(inputs, outputs)

    

model.compile(optimizer='Adam', loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
X_train, X_test, y_train, y_test = train_test_split(train_pad_seq_x, train_pad_seq_y, test_size=0.2, random_state=50)

for i in range(4):

    model.fit(X_train,y_train,batch_size=32,epochs=5,validation_data=(X_test,y_test),verbose=0)

    validation()

model.fit(train_pad_seq_x,train_pad_seq_y,batch_size=32,epochs=2,verbose=0)

validation()
test_pred = model.predict(train_pad_seq_x).round().astype(int)

avg = 0

for i in range(len(train_pad_seq_x)):  

    st1 = decode_sequence(train_pad_seq_x[i],test_pred[i])

    st2 = decode_sequence(train_pad_seq_x[i],train_pad_seq_y[i])

    avg += jaccard(st1,st2)

print("Jac all data sccore = " , np.sum(avg) / len(train_pad_seq_x))
sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

test_seq = token.texts_to_sequences(test.text)

test_pad_seq = pad_sequences(test_seq,maxlen=MAX_LEN)

test_pred = model.predict(test_pad_seq).round().astype(int)

for i in range(len(sub['selected_text'])):

    if test.sentiment[i] == 'neutral' or len(test.text[i].split()) < 4:

        sub['selected_text'][i] = test.text[i]

    else:

        sub['selected_text'][i] = decode_sequence(test_pad_seq[i],test_pred[i])

sub.to_csv('submission.csv', index=False)

sub.head()