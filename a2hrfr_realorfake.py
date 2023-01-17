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
df=pd.read_csv('../input/nlp-getting-started/train.csv')
df.head()
df.info()
len(df['location'].unique())
len(df['keyword'].unique())
df['text'].iloc[0]
df_neg=df[df['target']==0]
df_pos=df[df['target']==1]
df_neg.head()
!pip install pyspellchecker
import string

import re

from collections import Counter

from spellchecker import SpellChecker

spell = SpellChecker()

class CleanEn():



    def __init__(self):

        self.V = None



    def setupwords(self,wordlist,aslist=True):

        if type(wordlist) == str:

            wordlist = re.split(' |\n',wordlist)

        w = [] 

        for word in  wordlist:

            if word !='':

                

                word = word.strip()

                word = CleanEn.remove_URL(word)

                word = CleanEn.remove_emoji(word)

                word = CleanEn.remove_punct(word)

                word = CleanEn.remove_replace(word)

                #word = CleanEn.auto_corract(word)

                word = word.lower()

                

                

            if len(word.split(' ')) > 1:

                for i in word.split(' '):

                    w.append(i)

            elif len(word)  > 1 :

                w.append(word)    



        self.TheVoc(w)

        

        if aslist:

            return w

        return ' '.join(w)



    def TheVoc(self,final_words):

        if self.V == None:

            self.V = final_words

        else:

            self.V += final_words

            

    def get_voc(self):

        wordV = set(self.V)

        wordByCount = Counter(self.V)

        return wordV , wordByCount



    @staticmethod

    def remove_punct(text):

        if text =='':

            return text

        table=str.maketrans('','',string.punctuation)

        return text.translate(table)



    @staticmethod

    def auto_corract(text):

        if text =='':

            return text

        #misspelled_words = spell.unknown(text)

        return spell.correction(text)

 

        return text



    @staticmethod

    def remove_emoji(text):

        if text =='':

            return text

        emoji_pattern = re.compile("["

                               u"\U0001F600-\U0001F64F"  # emoticons

                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                               u"\U0001F680-\U0001F6FF"  # transport & map symbols

                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                               u"\U00002702-\U000027B0"

                               u"\U000024C2-\U0001F251"

                               u"\x89Û"

                               "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', text)



    @staticmethod

    def remove_URL(text):

        if text =='':

            return text

        url = re.compile(r'https?://\S+|www\.\S+')

        url2 = re.compile(r'https?://\S+|www\.\S+')

        t=url.sub(r'',text)

        t=url2.sub(r'',text)

        return t

    

    @staticmethod

    def remove_replace(text,n = 0):

        if n == 0 : 

            text = re.sub(r'^\d+','',text)

            #text = ' '.join(text)

            return CleanEn.remove_replace(text,n+1)

        

        elif n == 1:

            text = re.sub(r'^4','for ',text)

            #text = ' '.join(text)

            return CleanEn.remove_replace(text,n+1)

        

        elif n==2:

            t = re.findall('[a-z][^A-Z]*|[A-Z][^A-Z]*',text)

            if len(t) != len(text) :

                text = ' '.join(t)

            return  CleanEn.remove_replace(text,n+1)

            

        elif n==3:

            text = re.sub(r'[0-9]*','',text)

            return  CleanEn.remove_replace(text,n+1)

        

        elif n==5:

            text = re.sub(r'[0-9][^0-9]*','',text)

            return  CleanEn.remove_replace(text,n+1)

        

        else :

            return text

    

        







class Gram():



    def __init__(slef,words):

        assert type(words) == list

        assert len(words) != 0

        self.words=words 

    def n_gram(slef,N=2):

        if N <= len(self.words):

            return self.words

        w = [] 

        for i in range(len(self.words)- N):

            w.append(self.words[i:i+N])



        return w



    def uni_gram(slef):

        return self.words











cleanobject= CleanEn()

df['cleantext']=df['text'].apply(lambda x : cleanobject.setupwords(x,aslist=False))
df['cleantext'].head()
v , d= cleanobject.get_voc()
len(v)
max_len = 0

for i in df['cleantext']:

    if len(i.split(' ')) > max_len:

        max_len = len(i.split(' '))
max_len
by_count = {k: v for k, v in sorted(d.items(), key=lambda item: item[1],reverse=True)}
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
tok = Tokenizer()
tok.fit_on_texts(v)
seq = tok.texts_to_sequences(df['cleantext'].values)
X = pad_sequences(seq,maxlen=max_len)
voc_size= len(tok.index_word)
X.shape
len(df['target'].values)
def one_hot(val):

    s = len(val)

    arr=np.zeros((s,2))

    for c,v in enumerate(val):

        arr[c,v]=1

    return arr

y=one_hot(df['target'].values)
from keras.layers import Dense,LSTM,Dropout,SpatialDropout1D,Embedding

from keras.models import Sequential
def create_model():

    model = Sequential()

    model.add(Embedding(voc_size + 1, output_dim = 50, input_length = max_len))

    model.add(SpatialDropout1D(0.2))

    model.add(LSTM(132 , dropout=0.2, recurrent_dropout=0.2))

    model.add(Dropout(0.3))

    model.add(Dense(32))

    model.add(Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
model  = create_model()
def _k(k,i,X,y):

    n = X.shape[0]

    ipls = n // k



    ind1 = i * ipls

    ind2 = (i+1) * ipls

    XT = X[ind1 :ind2]

    YT = y[ind1 :ind2]



    xv= np.append(X[:ind1] , X[ind2:] ,axis =0)

    yv= np.append(y[:ind1] , y[ind2:] ,axis =0)



    return XT,YT,xv,yv
k = 5

for i in range(k):

    XT,YT,xv,yv = _k(k,i,X,y)

    hist = model.fit(XT,YT,epochs=10,batch_size=32,validation_data=[xv,yv])
test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
test_df.head()
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sample_sub.head()
test_df['cleantext'] = test_df['text'].apply(lambda x: cleanobject.setupwords(x))
testseq = tok.texts_to_sequences(test_df['cleantext'].values)
testseq = pad_sequences(testseq ,maxlen=max_len)
testseq.shape
pre = model.predict(testseq)

    

    

    
p = [np.argmax(x) for x in pre]
sample_sub['target'] = p
sample_sub.head()
sample_sub.to_csv('submission.csv',index=False)