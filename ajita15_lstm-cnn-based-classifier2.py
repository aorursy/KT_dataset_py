# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import os

import sys

from keras import backend as K

from keras.layers import Dense,Input, LSTM, Bidirectional, Embedding, TimeDistributed, SpatialDropout1D,GRU,CuDNNGRU,Dropout

from keras.layers import Conv1D,GlobalMaxPooling1D

from keras.preprocessing import text, sequence

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

from keras.models import Model

from keras.optimizers import Adam

from keras import losses

from keras import initializers as initializers, regularizers, constraints

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,roc_auc_score

import nltk

import re

from keras.engine.topology import Layer

# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")

print(df_train.shape)

print(df_test.shape)
train_X=df_train['comment_text']

test_X=df_test['comment_text']

train_Y=df_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values
df_train[df_train['threat']==1]['comment_text'].head()
from string import punctuation



contractions = {

"ain't": "am not","aren't": "are not","can't": "can not","can't've": "can not have","'cause": "because","could've": "could have",

"couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not",

"hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he had","he'd've": "he would have","he'll": "he will",

"he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is",

"i'd": "I would","i'd've": "I would have","i'll": "I shall / I will","i'll've": "I shall have / I will have","i'm": "I am",

"i've": "I have","isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is",

"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have",

"must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not","needn't've": "need not have",

"o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",

"shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have",

"she's": "she is","should've": "should have","shouldn't": "should not","shouldn't've": "should not have","so've": "so have",

"so's": "so is","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there would",

"there'd've": "there would have","there's": "there is","they'd": "they had","they'd've": "they would have","they'll": "they will",

"they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",

"we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not",

"what'll": "what will","what'll've": "what will have","what're": "what are","what's": "what is","what've": "what have",

"when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have","who'll": "who will",

"who'll've":"who will have","who's": "who is","who've": "who have","why's": "why is","why've": "why have","will've": "will have",

"won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not","wouldn't've": "would not have",

"y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

"you'd": "you had","you'd've": "you would have","you'll": "you will","you'll've": "you will have","you're": "you are","you've": "you have",

"&lt;3": " good ",":d": " good ",":dd": " good ",":p": " good ","8)": " good ",":-)": " good ", ":)": " good ",";)": " good ",

 "(-:": " good ","(:": " good ","yay!": " good ","yay": " good ","yaay": " good ","yaaay": " good ","yaaaay": " good ",

"yaaaaay": " good ",":/": " bad ",":&gt;": " sad ",":')": " sad ",":-(": " bad ",":(": " bad ", ":s": " bad ",":-s": " bad ",

"&lt;3": " heart ",":d": " smile ",":p": " smile ",":dd": " smile ","8)": " smile ", ":-)": " smile ", ":)": " smile ",

";)": " smile ","(-:": " smile ","(:": " smile ",":/": " worry ",":&gt;": " angry ", ":')": " sad ",":-(": " sad ",":(": " sad ",

":s": " sad ", ":-s": " sad ",r"\br\b": "are",r"\bu\b": "you",r"\bhaha\b": "ha",r"\bhahaha\b": "ha",r"\bdon't\b": "do not",

r"\bdoesn't\b": "does not",r"\bdidn't\b": "did not",r"\bhasn't\b": "has not",r"\bhaven't\b": "have not",r"\bhadn't\b": "had not",

r"\bwon't\b": "will not",r"\bwouldn't\b": "would not",r"\bcan't\b": "can not",r"\bcannot\b": "can not",r"\bi'm\b": "i am",

"m": "am","r": "are","u": "you","haha": "ha","hahaha": "ha","m": "am"}



def remove_punctuation(sent):

    l=[]

    for char in sent:

        if char not in punctuation:

            l.append(char)

        else:

            l.append(' ')

    return ''.join(l)



def lowerr(sent):

    l=[]

    doc=sent.split()

    for word in doc:

        if not word.islower():

            word=word.lower()

        if word[:4]=='http' or word[:3]=='www':

            continue

        if word in contractions.keys():

            word=contractions[word]

            l.extend(word.split())

        else:

            l.append(word)

    return ' '.join(l)



def remove_non_ascii(sent):

    return sent.encode('ascii', 'ignore').decode('ascii')



def remove_noise(input_text):

    text = re.sub('\(talk\)(.*)\(utc\)','',input_text)

    text = text.split()

    text = [re.sub('[\d]+','',x) for x in text]

    return ' '.join(text)
processed_sent=[]

for sent in train_X:

    sent=remove_non_ascii(sent)

    sent=lowerr(sent)

    sent=remove_noise(sent)

    sent=remove_punctuation(sent)

    processed_sent.append(sent)
processed_sent[:10]
processed_sent2=[]

for sent in test_X:

    sent=remove_non_ascii(sent)

    sent=lowerr(sent)

    sent=remove_noise(sent)

    sent=remove_punctuation(sent)

    processed_sent2.append(sent)
maxwords=100000

tok=text.Tokenizer(maxwords)

tok.fit_on_texts(list(train_X)+list(test_X))

train_X_sent=tok.texts_to_sequences(processed_sent)

test_X_sent=tok.texts_to_sequences(processed_sent2)

maxwords=min(maxwords,len(tok.word_index)+1)
df_train['token_comments']=train_X_sent

df_train['len_token_comments']=[len(sent) for sent in train_X_sent]

df_train['len_token_comments'].replace(0,np.nan,inplace=True)

print(np.sum(df_train.isnull()))
df2_train=df_train.dropna()

df2_train['len_token_comments']=[len(doc) for doc in df2_train['comment_text']]
print(max(df2_train['len_token_comments']))

print(min(df2_train['len_token_comments']))

print(np.mean(df2_train['len_token_comments']))
max_senten_len=400

train_X_sent=df2_train['token_comments']

train_X_sent=sequence.pad_sequences(train_X_sent,maxlen=max_senten_len)

test_X_sent=sequence.pad_sequences(test_X_sent,maxlen=max_senten_len)
train_Y=df2_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

#train_Y=np.reshape(train_Y,(train_Y.shape[0],1,train_Y.shape[1]))

print(train_Y.shape)
EMB_DIM=300

emb_layer=Embedding(maxwords,EMB_DIM,input_length=max_senten_len)

Inp=Input((max_senten_len,))

sent=emb_layer(Inp)

sent=Conv1D(128,5)(sent)

sent=GlobalMaxPooling1D()(sent)

#sent=Bidirectional(CuDNNGRU(128,return_sequences=False))(sent)

#sent=Dropout(rate=0.2)(sent)

sent=Dense(32,activation='relu')(sent)

pred=Dense(6,activation='sigmoid')(sent)

model=Model(Inp,pred)
train_X_sent.shape
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(train_X_sent,train_Y,epochs=5,validation_split=0.2,batch_size=128)
test_X_sent=np.asarray(test_X_sent)

y_pred=model.predict(test_X_sent,batch_size=1024,verbose=1)
df_sub=pd.read_csv("../input/sample_submission.csv")

df_sub[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]=y_pred
df_sub.to_csv("submission.csv",index=False)