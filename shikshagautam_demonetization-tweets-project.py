import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



tweets=pd.read_csv('../input/demonetization-in-india-twitter-data/demonetization-tweets.csv',encoding='ISO-8859-1')

tweets.info()
tweets['replyToSN']=tweets['replyToSN'].fillna('ArvindKejriwal')

tweets['replyToSID']=tweets['replyToSID'].fillna(8.010887e+17)

tweets['replyToUID']=tweets['replyToUID'].fillna(405427035.0)

tweets
xx=tweets['text']



yy=tweets['favorited']

xx

#tweets.info()

     
from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize

import nltk

from nltk.corpus import stopwords



stop_words=stopwords.words('english')

print(stop_words)

import string

def text_cleaning(xx):

    remove_punctuation = [char for char in xx if char not in string.punctuation]

 #print(remove_punctuation)

 # output is like 'h','e','l','l','o','','h','o','w','','a','r','e','','y','o','u'

    remove_punctuation=''.join(remove_punctuation)

# print(remove_punctuation)

# print(remove_punctuation.split())

 #output is like hello how are you

    return [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]

    print(word)

 # now we have cleaned title - no punctuation and stopword in it now

 
a=[]

for i in xx:

   

    text=" ".join(text_cleaning(i))

    print(text)

    a.append(text)

#a    
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(a,yy,test_size=.20,random_state=0)

from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()



yy=LE.fit_transform(yy)

yy

from sklearn.feature_extraction.text import CountVectorizer

vocabulary_count = CountVectorizer()

vocabulary_count



ss=vocabulary_count.fit_transform(x_train)

print(ss)

gg=vocabulary_count.transform(x_test)

ss=ss.toarray()

ss

gg=gg.toarray()

gg

from sklearn.naive_bayes import BernoulliNB

bn=BernoulliNB()

bn.fit(ss,y_train)

pred1=bn.predict(gg)

pred1

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred1))
from sklearn.naive_bayes import MultinomialNB

ml=MultinomialNB()

ml.fit(ss,y_train)

pred2=ml.predict(gg)

pred2

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred2))
from sklearn.feature_extraction.text import TfidfVectorizer

tf =TfidfVectorizer()

x_traint=tf.fit_transform(x_train)

x_testt=tf.transform(x_test)

print(x_traint)
xtf=x_testt.toarray()

xtrtf=x_traint.toarray()

xtf

xtrtf
from sklearn.naive_bayes import BernoulliNB

bnn=BernoulliNB()

bnn.fit(xtrtf,y_train)

pred3=bnn.predict(xtf)

pred3

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred3))
from sklearn.naive_bayes import MultinomialNB

mll=MultinomialNB()

mll.fit(xtrtf,y_train)

pred4=mll.predict(xtf)

pred4   

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,pred4))