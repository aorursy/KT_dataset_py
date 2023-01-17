import numpy as np 

import pandas as pd 

import os



import matplotlib.pyplot as plt

import matplotlib

import re

from textblob import TextBlob

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

import nltk

import seaborn as sns

data = pd.read_csv('../input/Tweets-BarackObama.csv')
data=data.rename(columns={'Tweet-text' : 'text'})
data.head(10)
data.sort_values('Likes', ascending=False)[['text','Likes']].head(5)
data.sort_values('Retweets', ascending=False)[['text','Retweets']].head(5)
data[['date','time']]=data.Date.str.split("_", expand = True)

data[['year','month','date']] = data.date.str.split("/", expand = True)
x=data['year'].value_counts()

x=x.sort_index()

sns.barplot(x.index,x.values,alpha=0.8)

matplotlib.rc('figure',figsize=[6,4])

plt.show()
txt = data['text'].str.lower().str.cat(sep=' ')

text1 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', txt)

text1 = re.sub('[^A-Za-z]+',' ' , text1)
words = nltk.tokenize.word_tokenize(text1)

word_dist = nltk.FreqDist(words)

stopwords = nltk.corpus.stopwords.words('english')

words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 



print('All frequencies, excluding STOPWORDS:')

print('=' * 40)

rslt = pd.DataFrame(words_except_stop_dist.most_common(10),

                    columns=['Word', 'Frequency']).set_index('Word')

print(rslt)

print('=' * 40)

matplotlib.style.use('ggplot')

matplotlib.rc('figure', figsize=[8,5])

rslt.plot.bar(rot=90)
data['polarity'] = data['text'].map(lambda text: TextBlob(text).sentiment.polarity)

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
data['polarity'].iplot(

    kind='hist', bins=20,

    xTitle='polarity',

    linecolor='black',

    yTitle='count',

    title='Sentiment Polarity Distribution')
data[data.polarity==-1].text.head(5)
data[data.polarity==0].text.head(5)
data[data.polarity==1].text.head(5)