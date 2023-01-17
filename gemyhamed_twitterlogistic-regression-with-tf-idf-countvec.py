# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import re

from nltk import word_tokenize

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

cols = ['sentiment','id','date','query_string','user','text']

df = pd.read_csv('../input/training.1600000.processed.noemoticon.csv', encoding='latin1', names=cols)
#knowing the data 

df.info()

df['sentiment'].value_counts()

df.head()

pat1 = '@[^ ]+'

pat2 = 'http[^ ]+'

pat3 = 'www.[^ ]+'

pat4 = '#[^ ]+'

pat5 = '[0-9]'



combined_pat = '|'.join((pat1, pat2, pat3, pat4, pat5))

# Cleaning 



clean_tweet_texts = []

for t in df['text']:

    t = t.lower()

    stripped = re.sub(combined_pat, '', t)

    tokens = word_tokenize(stripped)

    words = [x for x  in tokens if len(x) > 1]

    sentences = " ".join(words)

    negations = re.sub("n't", "not", sentences)

    

    clean_tweet_texts.append(negations)

clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])

clean_df['sentiment'] = df['sentiment'].replace({4:1})

clean_df.head()

clean_df.info()
neg_tweets = clean_df[clean_df['sentiment']==0]

pos_tweets = clean_df[clean_df['sentiment']==1]

# Getting the value count for every word

neg = neg_tweets.text.str.split(expand=True).stack().value_counts()

pos = pos_tweets.text.str.split(expand=True).stack().value_counts()



# Transforming to lists

values_neg = neg.keys().tolist()

counts_neg = neg.tolist()



values_pos = pos.keys().tolist()

counts_pos = pos.tolist()



plt.bar(values_neg[0:10], counts_neg[0:10])

plt.title('Top 10 Negative Words')

plt.show()



plt.bar(values_pos[0:10], counts_pos[0:10])

plt.title('Top 10 Positive Words')



plt.show()



cv = CountVectorizer(stop_words='english', binary=False, ngram_range=(1,1))



neg_cv = cv.fit_transform(neg_tweets['text'].tolist())

pos_cv = cv.fit_transform(pos_tweets['text'].tolist())

freqs_neg = zip(cv.get_feature_names(), neg_cv.sum(axis=0).tolist()[0])

freqs_pos = zip(cv.get_feature_names(), pos_cv.sum(axis=0).tolist()[0])

list_freq_neg = list(freqs_neg)

list_freq_pos = list(freqs_pos)
list_freq_neg.sort(key=lambda tup: tup[1], reverse=True)

list_freq_pos.sort(key=lambda tup: tup[1], reverse=True)

cv_words_neg = [i[0] for i in list_freq_neg]

cv_counts_neg = [i[1] for i in list_freq_neg]



cv_words_pos = [i[0] for i in list_freq_pos]

cv_counts_pos = [i[1] for i in list_freq_pos]

plt.bar(cv_words_neg[0:10], cv_counts_neg[0:10])

plt.xticks(rotation='vertical')

plt.title('Top Negative Words With CountVectorizer')

plt.show()



plt.bar(cv_words_pos[0:10], cv_counts_pos[0:10])

plt.xticks(rotation='vertical')

plt.title('Top Positive Words With CountVectorizer')

plt.show()



tv = TfidfVectorizer(stop_words='english', binary=False, ngram_range=(1,3))



neg_tv = tv.fit_transform(neg_tweets['text'].tolist())

pos_tv = tv.fit_transform(pos_tweets['text'].tolist())

freqs_neg_tv = zip(tv.get_feature_names(), neg_tv.sum(axis=0).tolist()[0])

freqs_pos_tv = zip(tv.get_feature_names(), pos_tv.sum(axis=0).tolist()[0])

list_freq_neg_tv = list(freqs_neg_tv)

list_freq_pos_tv = list(freqs_pos_tv)

list_freq_neg_tv.sort(key=lambda tup: tup[1], reverse=True)

list_freq_pos_tv.sort(key=lambda tup: tup[1], reverse=True)



cv_words_neg_tv = [i[0] for i in list_freq_neg_tv]

cv_counts_neg_tv = [i[1] for i in list_freq_neg_tv]



cv_words_pos_tv = [i[0] for i in list_freq_pos_tv]

cv_counts_pos_tv = [i[1] for i in list_freq_pos_tv]

plt.bar(cv_words_neg_tv[0:10], cv_counts_neg_tv[0:10])

plt.xticks(rotation='vertical')

plt.title('Top Negative Words With tf-idf')

plt.show()



plt.bar(cv_words_pos_tv[0:10], cv_counts_pos_tv[0:10])

plt.xticks(rotation='vertical')

plt.title('Top Positive Words with tf-idf')

plt.show()



x = clean_df['text']

y = clean_df['sentiment']

cv = CountVectorizer(stop_words='english', binary=False, ngram_range=(1,3))

x_cv = cv.fit_transform(x)

x_train_cv, x_test_cv, y_train_cv, y_test_cv = train_test_split(x_cv, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

log_cv = LogisticRegression() 

log_cv.fit(x_train_cv,y_train_cv)

from sklearn.metrics import confusion_matrix

y_pred_cv = log_cv.predict(x_test_cv)

print(confusion_matrix(y_test_cv,y_pred_cv))

from sklearn.metrics import classification_report

print(classification_report(y_test_cv,y_pred_cv))

tv = TfidfVectorizer(stop_words='english', binary=False, ngram_range=(1,3))

x_tv = tv.fit_transform(x)

x_train_tv, x_test_tv, y_train_tv, y_test_tv = train_test_split(x_tv, y, test_size=0.2, random_state=0)
log_tv = LogisticRegression() 

log_tv.fit(x_train_tv,y_train_tv)

y_pred_tv = log_tv.predict(x_test_tv)

print(confusion_matrix(y_test_tv,y_pred_tv))

print(classification_report(y_test_tv,y_pred_tv))
