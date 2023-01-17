# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

import string

import re

import os

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.cm as cm



from sklearn.cluster import MiniBatchKMeans

from subprocess import check_output

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize,RegexpTokenizer

from nltk.stem import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report,cohen_kappa_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.decomposition import TruncatedSVD 

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from gensim.corpora import Dictionary

from gensim.models import LdaModel

from collections import OrderedDict



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(check_output(["ls", "../input"]).decode("utf8"))



#tweets = pd.read_csv("../input/tweets_en_energy-nuclear_10-10_20-10.csv",sep=';',nrows=1000) #110000

tweets = pd.read_csv("../input/tweets-en-nuclear-all/tweets_en_nuclear_all.csv",sep=';',nrows=75000)

#tweets = pd.read_csv("../input/c04data/C04CSV.csv",quotechar="'",sep='@')

print(tweets.head())

print("Count=",tweets.count())
def tweets_success (row):

    if row['total_success'] == 0:

        return 0 #'no_success'

    if row['total_success']>0 and row['total_success']<10:

        return 1 #'average_success'

    if row['total_success']>=10 and row['total_success']<300:

        return 2 #'high_success'

    if row['total_success']>=300:

        return 3 #'splendid_success'

    

tweets['total_success'] = tweets['favorite_count'] + tweets['retweet_count']

tweets['success']=tweets.apply (lambda row: tweets_success(row), axis=1)



sns.countplot(x="success", data=tweets, palette="bwr")

plt.show()
import emoji



stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}',"`","--","’","!","—","nt","s","m",""])

stemmer = SnowballStemmer('english')



tweets['processed_text'] = tweets['text'].str.lower()

#tweets.head()



processed_tweets = []

tweets_text = tweets['processed_text'].values



for doc in tweets_text:

    doc = re.sub(r"(?:\@|#|https?)\S+", "", doc)

    doc = re.sub(r"(?![a-zA-Z])", "", doc)

    doc = re.sub(r"#|“|”|\?", "", doc)

    doc = re.sub(r"\s+", " ", doc)

    tokens = word_tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    no_emoji = [emoji.demojize(word) for word in filtered]

    stemmed = [stemmer.stem(word) for word in no_emoji]

    puntc = [re.sub(r":|[^a-zA-Z]", " ", word) for word in stemmed]

    space = [re.sub(r"\s+", "", word) for word in puntc]

    extra = [word for word in space if not re.match(r"^$|\s+|nt|s",word)]

    processed_tweets.append(extra)



tweets['processed_text']=processed_tweets





tweets.head()
row_lst = []

for lst in tweets.loc[:,'processed_text']:

    text = ''

    for word in lst:

        if not re.match(r"\s+|^$",word):

            text = text + ' ' + word

            

    row_lst.append(text)

    

tweets['processed_text_line'] = row_lst





tweets.head()

#pd.set_option('display.max_colwidth', -1)

#print(tweets.loc[[78]]['processed_text'])
tweets['processed_hashtags'] = tweets['text'].str.lower()

#tweets.head()



processed_tweets = []

tweets_text = tweets['processed_hashtags'].values



for doc in tweets_text:

    tokens = doc.split()

    hashtags = [word[1:] for word in tokens if re.match(r"#[A-Za-z0-9]+",word)]

    #processed_tweets.append(hashtags)

    

    text = ''

    row_lst = []

    for word in hashtags:

        text = text + ' ' + word

            

        #row_lst.append(text)

    processed_tweets.append(text)

    

tweets['processed_hashtags']=processed_tweets





tweets.head()
def tweets_negative (row):

    

    if re.search(r"danger|hazard|toxic|(toxic wast)|chernobyl|fukushima|hiroshima|nagasaki",row['processed_text_line']):

        return 1

    else:

        return 0

    

tweets['negativeTweet']=tweets.apply (lambda row: tweets_negative(row), axis=1)

tweets.head()
from textblob import TextBlob



def sentiment_score(df):

    text = df['processed_text_line']

    

    for i in range(0, len(text)):

        textB = TextBlob(text[i])

        sentiment_score = textB.sentiment.polarity

        df.set_value(i, 'sentiment_score', sentiment_score)

        

        

        if sentiment_score < 0.00:

            sentiment_class = 'Negative'

            df.set_value(i, 'sentiment_class', sentiment_class)

            

        elif sentiment_score > 0.00:

            sentiment_class ='Positive'

            df.set_value(i, 'sentiment_class', sentiment_class)

        else:

            sentiment_class = 'Notr'

            df.set_value(i, 'sentiment_class', sentiment_class)

    return df



sentiment_score(tweets)
tweets.to_csv('mycsvfile.csv',index=False)

#tweets.head()

#print(tweets['processed_text_line'])