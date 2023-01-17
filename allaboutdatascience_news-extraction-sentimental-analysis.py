!pip install newsapi-python

!pip install -U textblob



!python -m textblob.download_corpora

!pip install --upgrade pip
#Import the library

from newsapi import NewsApiClient

import json

import csv

import pandas as pd

import numpy as np
#list1=["Harvechem", "Wockhardt", "Bellen Chemistry", "Chempure","Nanjing Hoverchem", "3S International", "ABB Sciex PTE", "DHL","Novartis"]

list1=["DHL","Novartis"]



#Delete content of the csv file

#f = open('newsdata.csv', 'r+')

#f.truncate(0)



#open csv and write header

open("newsdata.csv", "w").close()

f = open("newsdata.csv", "w")

writer = csv.DictWriter(

    f, fieldnames=['Group','PublishedAt','Author','URL','Title','Description','Content'])

writer.writeheader()

f.close()


# Initiating the news api client with key

#newsapi = NewsApiClient(api_key='d3e1bf27692f43e290c0d1a6b192fd4e')

newsapi = NewsApiClient(api_key='6c0c2ea412374febbe8f03e6926bf06f')



for i in list1:

    #Using the client library to get articles related to search

    all_articles = newsapi.get_everything(q=i,

                                      language='en',

                                      from_param='2020-01-26',

                                      to='2020-02-05',

                                      sort_by ='relevancy')

    df = pd.DataFrame.from_dict(all_articles)

    df=df.assign(Group=i)

    df1= df.articles.apply(pd.Series)

    #Obtain data of id and name from source 

    df2= df1.source.apply(pd.Series)

    df3= pd.concat([df,df1,df2],axis=1).drop(['urlToImage','name','id','articles','source','status','totalResults'],1)

    df3=df3[['Group','publishedAt','author','url','title','description','content']]

    df3.to_csv('newsdata.csv',mode='a',header=False,index=False)

    #print(df3.head())



newsdata= pd.read_csv("newsdata.csv")

newsdata.head()
newsdata["combined"]= (newsdata["Title"]+newsdata["Description"]+newsdata["Content"])

newsdata.head()
pd.isnull(newsdata).sum()
newsdata = newsdata.dropna(axis=0, subset=['combined'])

pd.isnull(newsdata).sum()
newsdata['combined'].astype('str')
newsdata.dtypes
import nltk

#nltk.download()
#Data Preprocessing and Feature Engineering

from textblob import TextBlob

import re

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import string

punctuation=string.punctuation

stopwords=stopwords.words('english')

def _clean(text):

    text=text.lower()

    text= "".join(x for x in text if x not  in punctuation)

    

    words=text.split()

    words=[w for w in words if w not in stopwords]

    text=" ".join(words)

    

    return text

newsdata['cleaned']=newsdata['combined'].apply(_clean)

newsdata['cleaned']
def detect_polarity(text):

    return TextBlob(text).sentiment.polarity

newsdata['polarity'] = newsdata.cleaned.apply(detect_polarity)

newsdata.head()

#newsdata_final=newsdata[['Group','PublishedAt','Author','URL','Title','Description','Content', 'polarity']]

#newsdata_final.to_csv('newsdata_final.csv',mode='a',header=False,index=False)