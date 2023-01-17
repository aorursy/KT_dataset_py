import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.graph_objs as go

import plotly as py

import calendar

import re

import nltk

from nltk.corpus import stopwords

from textblob import TextBlob

from textblob import Word

import string

from gensim import corpora

from gensim.models.ldamodel import LdaModel

from gensim.parsing.preprocessing import preprocess_string

from gensim.models.coherencemodel import CoherenceModel

from collections import OrderedDict

import pyLDAvis.gensim
import pandas as pd

tweets_df = pd.read_csv("../input/elon-musks-tweets/data_elonmusk.csv",encoding='latin1')
tweets_df.head()
tweets_df=tweets_df.drop(['row ID','Retweet from','User'],axis=1)
tweets_df.head()
tweets_df['Time']=pd.to_datetime(tweets_df['Time'])
tweets_df['Time']=pd.to_datetime(tweets_df['Time'],format='%y-%m-%s %H:%M:%S')
tweets_df['Time']=pd.to_datetime(tweets_df['Time']).dt.to_period('M')
tweets_df['Time']=pd.DataFrame(tweets_df['Time'].astype(str))
tweets_df['Month']=tweets_df['Time'].apply(lambda x:x.split('-')[1]).astype(int)
tweets_df['Year']=tweets_df['Time'].apply(lambda x:x.split('-')[0])
tweets_df['Month']=tweets_df['Month'].apply(lambda x:calendar.month_name[x])
tweets_df['Year_month']=tweets_df['Year'].astype(str)+tweets_df['Month'].astype(str)
tweets_df=tweets_df.drop(['Month','Year','Time'],axis=1)
tweets_df.head()
HANDLE='@\w+'

LINK ='https://t\.co/\w+'



def basic_clean(text):

    text=re.sub(HANDLE,"",text)

    text=re.sub(LINK,"",text)

    

    return text
tweets_df['clean_tweet']=tweets_df['Tweet'].apply(lambda x:basic_clean(x))
tweets_df.head()
stops=stopwords.words('english')
tweets_df['clean_tweet']=tweets_df['clean_tweet'].apply(lambda x:" ".join(word.lower() for word in x.split() if word not in stops))
tweets_df['clean_tweet']=tweets_df['clean_tweet'].apply(lambda x:" ".join(Word(word).lemmatize() for word in x.split()))
retweet=['RT','rt','http']
punc=[string.punctuation]+retweet
tweets_df['clean_tweet']=tweets_df['clean_tweet'].apply(lambda x:" ".join(word for word in x.split() if word not in punc))
tweets_df.head()
tweets=tweets_df['clean_tweet'].apply(preprocess_string).tolist()
tweets
dictionary=corpora.Dictionary(tweets)
corpus=[dictionary.doc2bow(text) for text in tweets]
NUM_TOPICS=5

lda=LdaModel(corpus,num_topics=NUM_TOPICS,id2word=dictionary,passes=15)
lda.print_topics(num_words=6)
def calculate_coherence_score(tweets,dictionary,lda):

    coherence_model=CoherenceModel(model=lda,texts=tweets,dictionary=dictionary,coherence='c_v')

    return coherence_model.get_coherence()
def get_coherence_values(start,stop):

    for num_topics in range(start,stop):

        print(f'\nCalculating coherence for {num_topics} topics')

        lda=LdaModel(corpus,num_topics=num_topics,id2word=dictionary,passes=2)

        coherence=calculate_coherence_score(tweets,dictionary,lda)

        yield coherence
min_topics,max_topics=10,30

coherence_score=list(get_coherence_values(min_topics,max_topics))
x=[int(i) for i in range(10,30)]

plt.plot(x,coherence_score)
data={i:OrderedDict(lda.show_topic(i,27)) for i in range (NUM_TOPICS)}
data=pd.DataFrame(data)

data=data.fillna(0).T
print(data)
pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda,corpus,dictionary, mds='tsne')

panel