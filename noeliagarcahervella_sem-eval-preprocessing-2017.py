# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords 

from unicodedata import normalize

from nltk.stem import SnowballStemmer



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def csv_len(data) :

    

    words = []

    letters = []

    sentiments = []

    tweets = []

    

    for index, tweet in data.iterrows():

        tweet_split = tweet.Tweet.split()

        

        sentiments.append(tweet.Sentiment)

        tweets.append(tweet.Tweet)

        letters.append(len(tweet.Tweet))

        words.append(len(tweet_split))

    

    data['Tweet'] = tweets

    data['Sentiment'] = sentiments

    data['Words'] = words

    data['Letters'] = letters

    return data
def graphic(data_len) :

    

    fig,ax = plt.subplots(figsize=(5,5))

    plt.boxplot(data_len)

    plt.show()
def preprocessing(data) :

    

    tweets = []

    sentiment = []



    for index, tweet in data.iterrows():

        words_cleaned=""

        tweet_clean = tweet.Tweet.lower()

    

        words_cleaned =" ".join([word for word in tweet_clean.split()

            if 'http://' not in word

            and 'https://'not in word

            and '.com' not in word

            and '.es' not in word

            and not word.startswith('@')

            and not word.startswith('#')

            and word != 'rt'])

        

        

        tweet_clean = re.sub(r'\b([jh]*[aeiou]*[jh]+[aeiou]*)*\b',"",words_cleaned)

        tweet_clean = re.sub(r'(.)\1{2,}',r'\1',tweet_clean)

        tweet_clean = re.sub(

            r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 

            normalize( "NFD", tweet_clean), 0, re.I)

        tweet_clean = re.sub("[^a-zA-Z]"," ",tweet_clean)

        tweet_clean = re.sub("\t", " ", tweet_clean)

        tweet_clean = re.sub(" +", " ",tweet_clean) 

        tweet_clean = re.sub("^ ", "", tweet_clean)

        tweet_clean = re.sub(" $", "", tweet_clean)

        tweet_clean = re.sub("\n", "", tweet_clean)

        

        words_cleaned=""

        stemmed =""

        

        stop_words = set(stopwords.words('english'))

        stemmer = SnowballStemmer('english')

        

        tokens = word_tokenize(tweet_clean)

        

        words_cleaned =[word for word in tokens if not word in stop_words]

        stemmed = " ".join([stemmer.stem(word) for word in words_cleaned])

        

        

    

        sentiment.append(tweet.Sentiment)

        tweets.append(stemmed)

    

    data['Tweet'] = tweets

    data['Sentiment'] = sentiment

    data.loc[:,['Sentiment','Tweet']]

    

    return data
train = pd.read_csv('/kaggle/input/semevaldatadets/semeval-2017-train.csv', delimiter='	')

train.columns = ['Sentiment', 'Tweet']

train.rename(columns={'label': 'Sentiment','text' : 'Tweet'})
test = pd.read_csv('/kaggle/input/semevaldatadets/semeval-2017-test.csv', delimiter='	')

test.columns = ['Sentiment', 'Tweet']

test.rename(columns={'label': 'Sentiment','text' : 'Tweet'})
train.Sentiment.value_counts()
test.Sentiment.value_counts()
train = csv_len(train)

train
graphic(train['Words'])
graphic(train['Letters'])
test = csv_len(test)

test
graphic(test['Words'])
graphic(test['Letters'])
train_cleaned = preprocessing(train)

train_cleaned.loc[:,['Sentiment','Tweet']]
test_cleaned = preprocessing(test)

test_cleaned.loc[:,['Sentiment','Tweet']]
train_final = train_cleaned.loc[:,['Sentiment','Tweet']]

train_final

train_final.to_csv('semevalTrain.csv',index=False)
test_final = test_cleaned.loc[:,['Sentiment','Tweet']]

test_final

test_final.to_csv('semevalTest.csv',index=False)
train_cleaned = csv_len(train_cleaned)

train_cleaned
graphic(train_cleaned['Words'])
graphic(train_cleaned['Letters'])
test_cleaned = csv_len(test_cleaned)

test_cleaned
graphic(test_cleaned['Words'])
graphic(test_cleaned['Letters'])