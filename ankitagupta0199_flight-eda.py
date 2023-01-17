# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install wordcloud

import collections

import numpy as np

import pandas as pd

import matplotlib.cm as cm

import matplotlib.pyplot as plt

from matplotlib import rcParams

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline

import string





import re

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

import gensim 

from gensim.models import Word2Vec

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential



from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
os.chdir('/kaggle/input')



db=pd.read_csv("../input/flight-bookig/training_nlu_smaller.csv")

db.tail(10)
db.drop(['template'],axis=1,inplace=True)

db.loc[db['previous_intent']=='no','previous_intent',]=0



df = pd.DataFrame(columns=['query', 'current_intent'])

df
res1 = [] 

res2 = [] 

for i in range(0,len(db)):

    if db['previous_intent'][i]==0:

        res1.append(db['query'][i])

        res2.append(db['current_intent'][i])

df['query'] = res1

df['current_intent'] = res2 

df
df["current_intent"].value_counts()
all_headlines = ' '.join(df['query'].str.lower())

# all_headlines

stopwords = STOPWORDS



# stopwords.remove('no')

# stopwords.remove('not')



print(stopwords)

wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=1000).generate(all_headlines)
rcParams['figure.figsize'] = 10, 20

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# df.loc[df['current_intent']=='cancel','current_intent',]=0

# df.loc[df['current_intent']=='book','current_intent',]=1

# df.loc[df['current_intent']=='status','current_intent',]=2

# df.loc[df['current_intent']=='check-in','current_intent',]=3

# df.loc[df['current_intent']=='negation','current_intent',]=4



# df
def remove_punctuation(text):

    new_text=''.join([char for char in text if char not in string.punctuation])

    return new_text

df['new_text']=df['query'].apply(lambda row : remove_punctuation(row))

df.head()
def tokenize(text):

    tokens=re.split('\W+',text)

    return tokens 

df['tokenized_text']=df['new_text'].apply(lambda row : tokenize(row.lower()))

df.head()
ps = nltk.PorterStemmer()

def stemming(tokenized_text):

    stemmed_text=[ps.stem(word) for word in tokenized_text]

    return stemmed_text

df['stemmed_text']=df['tokenized_text'].apply(lambda row : stemming(row))

df[['query','stemmed_text']].head()
def get_final_text(stemmed_text):

    final_text=" ".join([word for word in stemmed_text])

    return final_text

df['final_text']=df.stemmed_text.apply(lambda row : get_final_text(row))

df.head()
df=df.drop('query',axis=1)

df=df.drop('new_text',axis=1)

df=df.drop('tokenized_text',axis=1)

df=df.drop('stemmed_text',axis=1)



df
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df['final_text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(stop_words=stopwords).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df['final_text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df['final_text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2),stop_words=stopwords).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df['final_text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df['final_text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3),stop_words=stopwords).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df['final_text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
from sklearn import preprocessing

Encode = preprocessing.LabelEncoder()

df['Label'] = Encode.fit_transform(df['current_intent'])
df['Label'].value_counts()
df.drop(['current_intent'],axis=1,inplace=True)
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(encoding = "latin-1", strip_accents = "unicode", stop_words = "english")

features = vec.fit_transform(df["final_text"])

print(features.shape)
from sklearn.model_selection import train_test_split

print(features.shape)

print(df["Label"].shape)

X_train, X_test, y_train, y_test = train_test_split(features, df["Label"], stratify = df["Label"], test_size = 0.2)