import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import json
!pip install vaderSentiment
## sentiment libraries

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime as dt

import time

today = dt.date.today()

yesterdays_date = '10-31-2017'#'{:%m-%d-%Y}'.format(dt.date(today.year, today.month-1, today.day))

print(yesterdays_date)

date = '10-31-2017'

with open('../input/October_tweets.txt', 'r') as f:

    tweets_d = json.loads(f.read())[date]

with open('../input/October.txt', 'r') as f:

    trends_d = json.loads(f.read())[date]

    sorted_trends = sorted(trends_d, key=trends_d.get, reverse=True)
def reorder(dat):

    '''

        given data in the format stored in the file - trend: {tweets: {tweet_text: count}...}

        

        reorder it to a list of the tweets - (tweet_text times the number of times they were originally seen). 

        so sum of original counts should equal len of new list --> 

                            

                            sum(dat[trend][tweets][count]) == len(new_dat[trend][tweets])        

        

        return dictionary of tweets ONLY ex. {trend: {tweets: [list of tweet texts]}}

        

    '''

    new_dat = {}

    for k in dat.keys():

        new_dat[k] = {}

        new_dat[k]['tweets'] = []

        t = [[x]*v for x,v in dat[k]['tweets'].items()]

        for i in t:

            for x in i:

                new_dat[k]['tweets'].append(x)

    return new_dat

## compound is overall measure of sentiment pos >=0.5, neg<=-0.5, neutral otherwise. 



## store top compound score (abs(compound))

top_compounder = {'x':0}



headers = ['pos', 'neg', 'neu', 'compound']

analyser = SentimentIntensityAnalyzer()

print('+'+'-'*81+'+')

print("|{:^25}        | {:^10}| {:^10}| {:^10}| {:^10}|".format('Trend','positive','negative','neutral','compound'))

print('|'+'-'*81+'|')

tweet_data = reorder(tweets_d)

for k in tweet_data.keys():

    tweets = tweet_data[k]['tweets']

    n = len(tweets)

    pos=compound=neu=neg=0

    for tweet in tweets:

        vs = analyser.polarity_scores(tweet)

        pos+=vs['pos']/n

        compound += vs['compound']/n

        neu += vs['neu']/n

        neg += vs['neg']/n

    if abs(compound) > abs(list(top_compounder.values())[0]):

        del top_compounder[list(top_compounder.keys())[0]]

        top_compounder[k] = compound

    if compound > 0.3:

        print('|{:<25} ({:^4}) | {:^10.5}| {:^10.5}| {:^10.5}| \x1b[6;30;42m{:^10.5}\x1b[0m|'.format(k[:25], n, pos, neg, neu, compound))

    elif compound < -.3:

        print('|{:<25} ({:^4}) | {:^10.5}| {:^10.5}| {:^10.5}| \x1b[3;30;41m{:^10.5}\x1b[0m|'.format(k[:25], n, pos, neg, neu, compound))

    else:

        print('|{:<25} ({:^4}) | {:^10.5}| {:^10.5}| {:^10.5}| {:^10.5}|'.format(k[:25], n, pos, neg, neu, compound))

#     break

print('+'+'-'*81+'+')
## sentiment for all trends

temp_all = [tweet_data[k]['tweets'] for k in tweet_data.keys()]

all_tweets = [j for i in temp_all for j in i]
n = len(all_tweets)

pos=compound=neu=neg=0

for tweet in all_tweets:

    vs = analyser.polarity_scores(tweet)

    pos+=vs['pos']/n

    compound += vs['compound']/n

    neu += vs['neu']/n

    neg += vs['neg']/n

if compound > 0.3:

    print('|{:<15} ({:^4}) | {:^10.5}| {:^10.5}| {:^10.5}| \x1b[6;30;42m{:^10.5}\x1b[0m|'.format('All Tweets', n, pos, neg, neu, compound))

elif compound < -.3:

    print('|{:<15} ({:^4}) | {:^10.5}| {:^10.5}| {:^10.5}| \x1b[3;30;41m{:^10.5}\x1b[0m|'.format('All Tweets', n, pos, neg, neu, compound))

else:

    print('|{:<15} ({:^4}) | {:^10.5}| {:^10.5}| {:^10.5}| {:^10.5}|'.format('All Tweets', n, pos, neg, neu, compound))
top_compounder
len(tweets)
# trend = np.random.choice(list(dat.keys()))

trend = list(top_compounder.keys())[0]
trend
print(len(set(tweet_data[trend]['tweets'])))

trending_tweets = set(tweet_data[trend]['tweets'])
from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import pprint

pp = pprint.PrettyPrinter(indent=4)
sum_sentences = []

candidate_tweets = {}

candidate_tweets_counts = {}

words = word_tokenize(''.join(trending_tweets))

stoppers = stopwords.words('english')+['RT', 'rt', 'https']

cleaned_words = [word.lower() for word in words

                 if (word.lower() not in stoppers and word.isalpha())]
word_freq = FreqDist(cleaned_words)

most_freq_words = word_freq.most_common(50)

pp.pprint(most_freq_words)
for tweet in set(trending_tweets):

    candidate_tweets[tweet] = tweet.lower()
for long, short in candidate_tweets.items():

    count = 0

    for freq_word, freq_score in most_freq_words:

        if freq_word in short:

            count += freq_score

    candidate_tweets_counts[long] = count
candidate_tweets_counts
from collections import OrderedDict

sorted_tweets = OrderedDict(sorted(candidate_tweets_counts.items(),

                                  key=lambda x: x[1],

                                  reverse=True)[:10])#top 10

print('\n\n'.join(set(sorted_tweets.keys())))
sorted_tweets.values()
most_freq_words
import matplotlib.pyplot as plt

%matplotlib inline

import nltk

from nltk.corpus import stopwords, PlaintextCorpusReader

# from nltk.book import *
import re

tweets_data = ''.join(set(tweet_data[trend]['tweets']))

tweets_data = re.sub('http\S+', '', tweets_data).replace('  ', ' ')

tweets_data = re.sub('@\S+', '.', tweets_data).replace(' . ', '. ')

tweets_data = tweets_data.replace('RT', '')

tweets_data = tweets_data.replace('\n', '')



print(tweets_data[:5000])
import gensim.summarization
summary = gensim.summarization.summarize(tweets_data, word_count=100)
print(summary)
# print(gensim.summarization.keywords(tweets_data))

def clean_tweets(tweets):

    import re

    tweets = set(tweets)

#     tweets = list(tweets.keys())

    tweet_str = ''.join(tweets)

    tweet_str = re.sub('http\S+', '', tweet_str).replace('  ', ' ') #removing urls

    tweet_str = re.sub('\S+.com?\S+', '', tweet_str)

    tweet_str = re.sub('@\S+', '.', tweet_str).replace(' . ', '. ') #removing mentions @

    tweet_str = re.sub('&\S+;', '', tweet_str).replace('  ',' ')#, tweet_str) #formatting

    tweet_str = tweet_str.replace('RT', ' ') #removing RT

    tweet_str = tweet_str.replace('\n\n', '')

    tweet_str = '.'.join(list(set(tweet_str.split('.'))))

#     print(tweet)

    

    return tweet_str.replace('  ',' ')
def get_summary(tweets, length_of_sum=100):

    import gensim.summarization

    return gensim.summarization.summarize(clean_tweets(tweets),

                                          word_count=length_of_sum)
for trend in sorted_trends[:10]:

    print(trend)

    print()

    if trend in tweet_data:

        print(get_summary(tweet_data[trend]['tweets'], 50))

    print()
from gensim import corpora

from gensim.models.ldamulticore import LdaMulticore

# from gensim.models.ldamodel import LdaModel

from gensim.parsing.preprocessing import STOPWORDS

from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords

import pprint
list(STOPWORDS.copy())

trends = np.random.choice(list(tweet_data.keys()), size=3, replace=False)
tweets = sent_tokenize(clean_tweets(tweet_data[trends[0]]['tweets'][:1000])+\

            clean_tweets(tweet_data[trends[1]]['tweets'][:1000])+\

            clean_tweets(tweet_data[trends[2]]['tweets'][:1000]))

STOPWORDS_ = set(list(STOPWORDS)+stopwords.words('spanish')+stopwords.words('english'))

texts1time = [[word for word in tweet.lower().split()

        if word not in STOPWORDS_ and word.isalnum()]

        for tweet in tweets]
len(texts1time)

# print(texts)
dictionary = corpora.Dictionary(texts1time)

corpus = [dictionary.doc2bow(text) for text in texts1time]
print(corpus[9])

print(texts1time[9])

print(dictionary[73])
num_topics = 3

passes=10

lda = LdaMulticore(corpus,

                   id2word=dictionary,

                   num_topics=num_topics,

                   passes=passes,

                   workers=3)
print(trends)
pp = pprint.PrettyPrinter(indent=4)

pp.pprint(lda.print_topics(num_words=10))
from operator import itemgetter

lda.get_document_topics(corpus[33],minimum_probability=0.05,per_word_topics=False)

sorted(lda.get_document_topics(corpus[33],minimum_probability=0,per_word_topics=False),key=itemgetter(1),reverse=True)
print(texts1time[33])
def draw_wordcloud(lda,topicnum,min_size=0,STOPWORDS=[]):

    from nltk.corpus import stopwords

    STOPWORDS += stopwords.words('spanish')

    word_list=[]

    prob_total = 0

    for word,prob in lda.show_topic(topicnum,topn=50):

        prob_total +=prob

    for word,prob in lda.show_topic(topicnum,topn=50):

        if word in STOPWORDS or  len(word) < min_size:

            continue

        freq = int(prob/prob_total*1000)

        alist=[word]

        word_list.extend(alist*freq)



    from wordcloud import WordCloud, STOPWORDS

    import matplotlib.pyplot as plt

    %matplotlib inline

    text = ' '.join(word_list)

    wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white',width=3000,height=3000).generate(' '.join(word_list))





    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
for i in range(3):

    draw_wordcloud(lda,i)
len(corpus)
tweets = [clean_tweets(list(set(tweet_data[k]['tweets']))) for k in dat.keys()]

texts = [[word for word in tweet.lower().split()

        if word not in STOPWORDS_ and word.isalnum()]

        for tweet in tweets]



dictionary_by_trend = corpora.Dictionary(texts)

text_list = []

for i in range(len(tweets)):

    doc = []

    for word in texts[i]:

        if word in STOPWORDS_ or not word.isalpha() or len(word)<5:

            continue

        doc.append(word)

    text_list.append(doc)

by_trend_corpus = [dictionary_by_trend.doc2bow(text) for text in text_list]
# print(len(tweets[0]))

len(by_trend_corpus)
lda_by_trend = LdaMulticore(by_trend_corpus,

                           id2word=dictionary_by_trend,

                           num_topics=20,

                           passes=10,

                           workers=3)
pp = pprint.PrettyPrinter(indent=4)

pp.pprint(lda_by_trend.print_topics(num_words = 10))
from operator import itemgetter

sorted(lda_by_trend.get_document_topics(by_trend_corpus[1],minimum_probability=0,per_word_topics=False),key=itemgetter(1),reverse=True)
text_list[1]
draw_wordcloud(lda_by_trend, 1)
print(lda_by_trend.show_topic(1, topn=5))

print(lda_by_trend.show_topic(7, topn=5))
from gensim.similarities.docsim import Similarity

from gensim import corpora, models, similarities

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

doc = """

The US Open has been a real doozy. Federer has been on the ropes and Nadal was alomst take out as well. New york is in for a good tournement. Marin cilic is looking like the most comfortable player out here.

"""

vec_bow = dictionary.doc2bow(doc.lower().split())

vec_lsi = lsi[vec_bow]

index = similarities.MatrixSimilarity(lsi[corpus])

sims = index[vec_lsi]

sims = sorted(enumerate(sims), key=lambda item: -item[1])

sims
# print(corpus[507])

# print(dictionary[170])

texts1time[sims[0][0]]