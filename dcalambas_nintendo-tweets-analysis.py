import sys

import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from nltk.corpus import stopwords

from sklearn import preprocessing

import json

import string

import os

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import nltk

tweets = []

for line in open('../input/NintendoTweets.json', 'r'):

    if len(line)>1:

        tweets.append(json.loads(line))
def clean_text(text, stop_w):

    text = text.lower()

    text = text.replace('@',' ')

    text = text.replace('#',' ')

    text = text.replace('.',' ')

    text = text.replace(',',' ')  

    text = text.replace('\n',' ')

    text = text.replace('!',' ')

    text = text.replace('?',' ')

    text = text.replace('¡',' ')

    text = text.replace('¿',' ')

    text = text.replace(':',' ')

    return text



cant_inicial=len(tweets)

for t in tweets:

    if 'text' in t:

        t['text'] = clean_text(t['text'], stop_w)

    else:

        #print("NO TIENE TEXTO: "+str(t))

        tweets.remove(t)

cant_final=len(tweets)



print(str(cant_inicial) + " -> "+ str(cant_final))


complete_text = ''





for t in tweets:

    if 'text' in t:

        tokens = t['text'].split(' ')

        for token in tokens:

            complete_text += ' '+token



wordcloud = WordCloud(width = 800, height = 800,

                background_color ='white',

                stopwords = stopwords.words('english')+['http','co','rt','https','//t','``','sm',"n't",'c','p','r','(',')','#','m','h',"'m","'","'s",'f','…','sh','n'],

                min_font_size = 10).generate(complete_text)



# plot the WordCloud image                       

plt.figure(figsize = (8, 8), facecolor = None)

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout(pad = 0)

 

plt.show()
tweets_words = nltk.tokenize.word_tokenize(complete_text)

tweets_filtered=[word for word in tweets_words if word not in stopwords.words('english')+['http','co','rt','https','//t','``','sm',"n't",'c','p','r','(',')','#','m','h',"'m","'","'s",'f','…','sh','n']]

tweets_tokens = nltk.Text(tweets_filtered)

freqrp=nltk.FreqDist(tweets_tokens)

top20 = freqrp.most_common(20)

print("TOP 20 COMUNES: "+ str(top20))

import numpy as np



top20_tuples = [(v, k) for k, v in top20]

x, y = zip(*top20_tuples)

print(x)

y_pos = np.arange(len(x))



plt.barh(y, x, align='center', alpha=0.5)

plt.yticks(y_pos, y)

plt.xlabel('Cantidad de ocurrencias')

plt.title('Top 20 palabras usadas')

plt.show()
import re

import operator



source = {}

for t in tweets:

    if 'source' in t:

        s = t['source']

        result = re.search(r'\">(.*)<\/', s)

        if result.group(1) in source:

            source[result.group(1)] += 1

        else:

            source[result.group(1)] = 1

           

    

source_top = dict(sorted(source.items(), key=operator.itemgetter(1), reverse=True)[:5])

print(source_top)
source_top_tuples = [(v, k) for k, v in source_top.items()]



x, y = zip(*source_top_tuples)

y_pos = np.arange(len(x))



plt.barh(y, x, align='center', alpha=0.5)

plt.yticks(y_pos, y)

plt.xlabel('Source')

plt.title('E3 2018 Nintendo Conference - tweets source')

plt.show()
text_x_minute = {}



for t in tweets:

    if 'created_at' in t:

        t_date = t['created_at']

        result = re.search(r':(.*):', t_date)

        if result.group(1) not in text_x_minute:

            text_x_minute[result.group(1)] = ''

        tokens = t['text'].split(' ')

        for token in tokens:

            text_x_minute[result.group(1)] += ' '+token

!{sys.executable} -m pip install sentic

!{sys.executable} -m pip install senticnet
from sentic import SenticPhrase



text_x_minute_sentiment = {}



for txm in text_x_minute.keys():

    sp = SenticPhrase(text_x_minute[txm])

    sp_sentiment=sp.get_sentiment()

    if 'positive' in sp_sentiment:

        text_x_minute_sentiment[txm]=1

    if 'negative' in sp_sentiment:

        text_x_minute_sentiment[txm]=-1

    if 'neutral' in sp_sentiment:

        text_x_minute_sentiment[txm]=0



print(text_x_minute_sentiment)
        

text_x_minute_top = {}

for txm in text_x_minute.keys():

    tweets_words = nltk.tokenize.word_tokenize(text_x_minute[txm])

    tweets_filtered=[word for word in tweets_words if word not in stopwords.words('english')+['http','co','rt','https','//t','``','sm',"n't",'c','p','r','(',')','#','m','h',"'m","'","'s",'f','…','sh','n']]

    tweets_tokens = nltk.Text(tweets_filtered)

    freqrp=nltk.FreqDist(tweets_tokens)

    top_word = freqrp.most_common(1)

    text_x_minute_top[txm] = top_word



print(text_x_minute_top)