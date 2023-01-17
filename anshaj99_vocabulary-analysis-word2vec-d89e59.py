# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load required libraries

import numpy as np

import pandas as pd

#For displaying complete rows info

pd.options.display.max_colwidth=500

import tensorflow as tf

import spacy

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from collections import Counter

print(tf.__version__)
#Load data into pandas dataframe

df=pd.read_csv("../input/news_summary.csv",encoding="latin1")

print(df["text"][0],"\n",df["ctext"][0])
#Properly formatted data removing nans

df.drop_duplicates(subset=["ctext"],inplace=True)

df.dropna(inplace=True)

df.reset_index(drop=True,inplace=True)
df.describe()
#Using a Generator

def readline(data):

    for i in range(len(data)):

        yield data[i]

stopWords = stopwords.words('english')
headlines=readline(df["headlines"])

nlp = spacy.load('en')

hvocab=[]

for i,headline in enumerate(headlines):

    #print (i)

    headline=headline.lower()

    for token in nlp.tokenizer(headline):

        if len(token)>2:

            hvocab.append(str(token))

print(len(hvocab),len(set(hvocab)))

hvocab=list(set(hvocab))
headlines=readline(df["headlines"])

nltk_hvocab=[]

for i,headline in enumerate(headlines):

    #print (i)

    headline=headline.lower()

    for token in word_tokenize(headline):

        if len(token)>2:

            nltk_hvocab.append(token)

print(len(nltk_hvocab),len(set(nltk_hvocab)))

nltk_hvocab=list(set(nltk_hvocab))
text_vocab=[]

for i,text in enumerate(df["text"]):

    #print (i)

    text=text.lower()

    for token in word_tokenize(text):

        if len(token)>2:

            text_vocab.append(token)

print(len(text_vocab),len(set(text_vocab)))

text_vocab=list(set(text_vocab))
ctext_vocab=[]

for i,ctext in enumerate(df["ctext"]):

    #print (i)

    ctext=ctext.lower()

    ctext=ctext.replace("."," ")

    for token in word_tokenize(ctext):

        if len(token)>2:

            ctext_vocab.append(token)

print(len(ctext_vocab),len(set(ctext_vocab)))

ctext_vocab=list(set(ctext_vocab))
# Check for Matching vocabulary 
count=0

for text in text_vocab:

    if text in ctext_vocab:

        count=count+1

#Percentage of words that match between summary and complete article.

print((count/len(text_vocab))*100)
import gensim

import string

import re

articles_tokens=[]

for i in range(len(df["ctext"])):

    articles_tokens.append([word for word in word_tokenize(df["ctext"][i].lower().replace("."," ").translate(string.punctuation)) if len(word)>2])



#model = gensim.models.Word2Vec(sentence, min_count=1,size=100,workers=4)
model = gensim.models.Word2Vec(articles_tokens, min_count=5,size=100,workers=4)
for i,article in enumerate(articles_tokens):

    for j,token in enumerate(article):

        if len(token)>2 and token not in stopWords:

            articles_tokens[i][j]=re.sub(r'[0-9]+',"NUM",token)
model.wv.most_similar("rape")