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
from fastai.text import *

import html
import nltk

# import nltk

# nltk.download('all', halt_on_error=False)
from nltk.tokenize import sent_tokenize, word_tokenize
BOS = 'xbos'  # beginning-of-sentence tag

FLD = 'xfld'  # data field tag
data_file_path = "../input/internship_takeHome_data.csv"
#Library Imports

import numpy as np 

import pandas as pd 

import bz2

import gc

import chardet

import re

import os

from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

import scipy.stats as stats



from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.manifold import TSNE
#Keras Imports

from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPool1D, Dropout, concatenate, Layer, InputSpec, CuDNNLSTM

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras import activations, initializers, regularizers, constraints

from keras.utils.conv_utils import conv_output_length

from keras.regularizers import l2

from keras.constraints import maxnorm
#Read Train File

file_data_initial=pd.read_csv(data_file_path)
file_data=file_data_initial.copy()

file_data.head()
temp=file_data['"id;""polarity"";""tweet""",,,,,,,'].iloc[0]
def get_reg_exp(x,i):

    x=re.sub('""','',x)

    x=re.sub('"','',x)

    x=re.sub(',','',x)

#     x=x[1:]

    x=x.split(';')

    return x[i]
file_data["ID"]=file_data.apply(lambda x: get_reg_exp(x['"id;""polarity"";""tweet""",,,,,,,'], 0), axis=1)

file_data["Polarity"]=file_data.apply(lambda x: get_reg_exp(x['"id;""polarity"";""tweet""",,,,,,,'], 1), axis=1)

file_data["Tweet"]=file_data.apply(lambda x: get_reg_exp(x['"id;""polarity"";""tweet""",,,,,,,'], 2), axis=1)
file_data.head()
file_data.to_csv('cleaned_data.csv',index=False)
file_data["Polarity"].value_counts()
file_data["Tweet"].nunique()
tweets=file_data["Tweet"].tolist()

tweets[0]
allwords=[]

for i in tweets:

    tokens=i.split()

    for j in tokens:

        allwords.append(j)
len(allwords)
subs="@"

res = [i for i in allwords if subs in i]
import collections

counter=collections.Counter(res)
sortedres = sorted(counter, key=counter.get, reverse=True)

most_tweeted={}

for r in sortedres:

    most_tweeted[r]=counter[r]

del most_tweeted['@']
most_tweet=pd.DataFrame(list(most_tweeted.items()), columns=['Twitter Account', 'Times Mentioned'])

most_tweet.head()
# most_tweet.plot(x='Twitter Account', y='Times Mentioned', style='bar')

plt.bar(most_tweet[:10]['Twitter Account'], most_tweet[:10]['Times Mentioned'])

plt.show()
def get_top_n_words(n_top_words, count_vectorizer, text_data):

    '''

    returns a tuple of the top n words in a sample and their 

    accompanying counts, given a CountVectorizer object and text sample

    '''

    vectorized_headlines = count_vectorizer.fit_transform(text_data)

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)

    word_values = np.flip(np.sort(vectorized_total)[0,:],1)

    

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))

    for i in range(n_top_words):

        word_vectors[i,word_indices[0,i]] = 1



    words = [word[0].encode('ascii').decode('utf-8') for 

             word in count_vectorizer.inverse_transform(word_vectors)]



    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer=CountVectorizer('english')

words, word_values = get_top_n_words(n_top_words=10,  

                                     count_vectorizer=count_vectorizer, 

                                     text_data=tweets)
tempdf=pd.DataFrame()
tempdf["words"]=words

tempdf["freq"]=word_values
tempdf
train_data=pd.read_csv("../input/train.csv", encoding='latin-1')

train_data.head()
# Sentences and their sentiments

sent=train_data["SentimentText"].tolist()

train_y=train_data["Sentiment"].tolist()
for i in range(len(sent)):

    sent[i] = re.sub('  ','',sent[i])
#Some more preprocessing

for i in range(len(sent)):

    if 'www.' in sent[i] or 'http:' in sent[i] or 'https:' in sent[i] or '.com' in sent[i]:

        sent[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", sent[i])      
max_features = 10000

maxlen = 100

max_len=100
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(sent)
tokenized_train = tokenizer.texts_to_sequences(sent)

X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
X_train[0]
batch_size = 32

epochs = 8
def cudnnlstm_model(conv_layers = 2, max_dilation_rate = 3):

    embed_size = 128

    inp = Input(shape=(maxlen, ))

    x = Embedding(max_features, embed_size)(inp)

    x = Dropout(0.25)(x)

    x = Conv1D(2*embed_size, kernel_size = 3)(x)

    prefilt = Conv1D(2*embed_size, kernel_size = 3)(x)

    x = prefilt

    for strides in [1, 1, 2]:

        x = Conv1D(128*2**(strides), strides = strides, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_size=3, kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)

    x_f = CuDNNLSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)  

    x_b = CuDNNLSTM(512, kernel_regularizer=l2(4e-6), bias_regularizer=l2(4e-6), kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10))(x)

    x = concatenate([x_f, x_b])

    x = Dropout(0.5)(x)

    x = Dense(64, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['binary_accuracy'])



    return model
cudnnlstm_model = cudnnlstm_model()

cudnnlstm_model.summary()
weight_path="early_weights.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks = [checkpoint, early_stopping]
cudnnlstm_model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, shuffle = True, validation_split=0.15, callbacks=callbacks)
for i in range(len(tweets)):

    if 'www.' in tweets[i] or 'http:' in tweets[i] or 'https:' in tweets[i] or '.com' in tweets[i]:

        tweets[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", tweets[i])
tokenized_train = tokenizer.texts_to_sequences(tweets)

X_test = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
predicted=cudnnlstm_model.predict(X_test)
predicted
result_binary = []

for output in predicted:

    if output < 0.5573:

        result_binary.append("4")

    else:

        result_binary.append("0")
count=0

for i in result_binary:

    if(i=='4'):

       count=count+1

print(count)
result_binary[77:100]