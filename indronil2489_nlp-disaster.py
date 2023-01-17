import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_tweet=df_train['text']

y_train=df_train['target']

test_tweet=df_test['text']
import re

from string import punctuation

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):

    stemmer = PorterStemmer()

    stopword = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)

    tweet = re.sub(r'^RT[\s]+', '', tweet)

    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = ""

    for word in tweet_tokens:

        if (word not in stopword and word not in punctuation):

            stem_word = stemmer.stem(word)

            tweets_clean=tweets_clean+stem_word+" "

    return tweets_clean
tweets=[]

for i in range(0,len(train_tweet)):

    tweets.append(process_tweet(train_tweet[i]))
t_tweets=[]

for i in range(0,len(test_tweet)):

    t_tweets.append(process_tweet(test_tweet[i]))
t_tweets
from sklearn import feature_extraction

count_vectorizer = feature_extraction.text.TfidfVectorizer()

x_train = count_vectorizer.fit_transform(tweets)

x_test = count_vectorizer.transform(t_tweets)
x_train=x_train.toarray()

x_test=x_test.toarray()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1.0, penalty='l2', max_iter=500) 

lr.fit(x_train, y_train)

y_test=lr.predict(x_test)
from sklearn.ensemble import RandomForestClassifier



rforest = RandomForestClassifier(n_estimators = 1000, random_state = 1) 

rforest.fit(x_train,y_train)

y_test = rforest.predict(x_test)
y_test
id_test =df_test['id']
df3 = pd.DataFrame()

df3['id'] = id_test.tolist()

df3['target'] = y_test.tolist()

df3.to_csv("./file.csv", sep=',',index=True)
df3.target.value_counts().plot(kind = 'bar')