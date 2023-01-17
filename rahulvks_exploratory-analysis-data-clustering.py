import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import wordcloud

import nltk

%matplotlib inline

from datetime import datetime

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import re

from string import punctuation

stop = set(stopwords.words('english'))

from collections import Counter

import datetime as dt

import plotly.plotly as py

import plotly.graph_objs as go

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.summarization import summarize

from gensim.models import word2vec

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

Meta_Data = pd.read_csv("../input/inaug_speeches.csv",encoding='iso-8859-1')



Meta_Data.head(1)
Presidents =Meta_Data.groupby(['Name']).size()

Presidents.plot(kind="bar",ecolor='r', align='center')
def cleaning(s):

    s = str(s)

    s = s.lower()

    s = re.sub('\s\W',' ',s)

    s = re.sub('\W,\s',' ',s)

    s = re.sub(r'[^\w]', ' ', s)

    s = re.sub("\d+", "", s)

    s = re.sub('\s+',' ',s)

    s = re.sub('[!@#$_]', '', s)

    s = s.replace("co","")

    s = s.replace("https","")

    s = s.replace(",","")

    s = s.replace("[\w*"," ")

    s = re.sub("mr"," ",s)

    return s

Meta_Data['text'] = [cleaning(s) for s in Meta_Data['text']]

Meta_Data['text'] = Meta_Data.apply(lambda row: nltk.word_tokenize(row['text']),axis=1)

Meta_Data['text'] = Meta_Data['text'].apply(lambda x : [item for item in x if item not in stop])

  
President_Name = Meta_Data['Name']

Speech_Text = Meta_Data['text']

Date = Meta_Data['Date']
for i in range(len(Meta_Data)):

    print(President_Name[i],'No of words',len(Speech_Text[i]),'Date',Date[i])
Speech_Date=Meta_Data.groupby(['Name','Date']).size()

Speech_Date
text = Meta_Data['text']

words = pd.Series(' '.join(Meta_Data['text'].astype(str)).lower().split(" ")).value_counts()[:50]

A=words.plot()