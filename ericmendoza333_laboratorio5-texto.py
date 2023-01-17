

import numpy as np

import pandas as pd

import re

from collections import Counter

from nltk.corpus import stopwords

from PIL import Image

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.util import ngrams

from nltk import word_tokenize

from nltk.corpus import reuters

from collections import Counter, defaultdict

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
f = open('/kaggle/input/tweets-blogs-news-swiftkey-dataset-4million/final/en_US/en_US.twitter.txt', 'r')

tweets = f.readlines()

f.close()



f = open('/kaggle/input/tweets-blogs-news-swiftkey-dataset-4million/final/en_US/en_US.news.txt', 'r')

news = f.readlines()

f.close()



f = open('/kaggle/input/tweets-blogs-news-swiftkey-dataset-4million/final/en_US/en_US.blogs.txt', 'r')

blogs = f.readlines()

f.close()

tweets= [i.lower() for i in tweets]

news= [i.lower() for i in news]

blogs= [i.lower() for i in blogs]

tweets
tweets=[re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','',i) for i in tweets]

news=[re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','',i) for i in news]

blogs=[re.sub('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','',i) for i in blogs]
tweets=[re.sub('[^a-zA-Z0-9 ]+','', i) for i in tweets]

news=[re.sub('[^a-zA-Z0-9 ]+', '', i) for i in news]

blogs=[re.sub('[^a-zA-Z0-9 ]+', '', i) for i in blogs]

news
tweets
blogs
stopwords=set(stopwords.words('english'))

trendy_words=['im','rt','jk','btw','lol','yolo','lmao','lmfao','fb','like','get','em']

for i in trendy_words:

    stopwords.add(i)



        

    
tweets_clean=[]

for tweet in tweets:

    word_list=[]

    for word in tweet.split():

        if word not in stopwords:

            word_list.append(word)

    tweets_clean.append(' '.join(word_list))

tweets_clean
blogs_clean=[]

for blog in blogs:

    word_listo=[]

    for word in blog.split():

        if word not in stopwords:

            word_listo.append(word)

    blogs_clean.append(' '.join(word_listo))

blogs_clean
news_clean=[]

for new in news:

    list_words=[]

    for word in new.split():

        if word not in stopwords:

            list_words.append(word)

    news_clean.append(' '.join(list_words))

news_clean
tweet_count=Counter()



for tweet in tweets_clean:

    tweet_count.update(palabra.strip('.,?!"\'').lower() for palabra in tweet.split())

tweet_count
news_count=Counter()



for title in news_clean:

    news_count.update(palabra.strip('.,?!"\'').lower() for palabra in title.split())

news_count
blog_count =Counter()

for blog in blogs_clean:

    blog_count.update(palabra.strip('.,?!"\'').lower() for palabra in blog.split())

blog_count
tweet_count.most_common(5)
blog_count.most_common(5)
news_count.most_common(5)
wc = WordCloud(background_color="white",width=1000,height=1000, max_words=20,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(tweet_count)

plt.imshow(wc)
wc = WordCloud(background_color="white",width=1000,height=1000, max_words=20,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(blog_count)

plt.imshow(wc)
wc = WordCloud(background_color="white",width=1000,height=1000, max_words=20,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(news_count)

plt.imshow(wc)
#uniendo las listas

master_frame=tweets_clean+blogs_clean+news_clean

master_frame
val=round(len(master_frame)*0.1,0)

random_sample=random.sample(master_frame,int(val))

random_sample
import warnings



warnings.filterwarnings('ignore')

two_gram=[]

size=2

for i in range(len(random_sample)):

    for item in ngrams(random_sample[i].split(),size):

        two_gram.append(item)

print(len(two_gram))

two_gram
three_gram=[]

size=3

for i in range(len(random_sample)):

    for item in ngrams(random_sample[i].split(),size):

        three_gram.append(item)

print(len(three_gram))

three_gram
model = defaultdict(lambda: defaultdict(lambda: 0))

for i,j,k in three_gram:

    model[(i,j)][k] +=1

for i,j in model:

    total=float(sum(model[(i,j)].values()))

    for k in model[(i,j)]:

        model[(i,j)][k] /= total

        

dict(model)