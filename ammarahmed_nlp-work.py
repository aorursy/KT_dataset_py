# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

"""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""



"""

Created on Sat Dec 14 22:50:37 2019



@author: DELL inspiron

"""



import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 



tokenizer = nltk.tokenize.RegexpTokenizer('\w+')



#with some help from

#https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python

#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

#https://www.programiz.com/python-programming/examples/remove-punctuation



#read in the file

df = pd.read_csv(r'/kaggle/input/tweets/tweets.csv',encoding = "ISO-8859-1")

stop_words=set(stopwords.words('english'))

#stop_words= ['ourselves','hers','between','yourself','but','again','there','about','once','during','out']

punctuations = list('''!()-[]{};:'"\,<>./?@#$%^&*_~''')

extra_filters=['https','@']
#df2=df.iloc[:10,:]

df2=df



#convert the file to text

list_of_tweets=list(df2.text)



#join each row's text in one text body

text=' '.join(list_of_tweets)



#tokenize it

word_tokens = word_tokenize(text)



#filter stopwords, punctuations and some other non-useful text like 'https','@' since we are analyzing tweets

filtered_text=[a for a in word_tokens if a not in stop_words and a not in punctuations and a not in extra_filters]



#take the frequency distribution

freqdist=nltk.FreqDist(filtered_text)



#check whether ranking is done ok

freqdist_dict=dict(freqdist)

ranked=list(reversed(sorted(freqdist_dict, key=freqdist_dict.get)))

for b in ranked[:3]:

    print(b,freqdist_dict[b])
#now plot the top 10

plot = freqdist.plot(10)

#tagged = nltk.pos_tag(word_tokenize(list_of_tweets[0]))

#tagged



#POS tag it, store it in a new column and check.

df2['POS']=df2['text'].apply(lambda x: nltk.pos_tag(word_tokenize(x)))

df2.head()