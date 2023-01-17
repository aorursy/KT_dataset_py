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
train = pd.read_csv('../input/train.csv')
#Nomber of words
train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
train[['tweet','word_count']].head()
# Number of charecters
train['char_count'] = train['tweet'].str.len()
train[['tweet','char_count']].head()
#Average word length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','avg_word']].head()
# Number of stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()
#Number of special charecters

train['hastags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hastags']].head()
#Numberof numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()
#Number of uppercase words
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()
#transform tweets into Lower case
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()
#Removing punctuation
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()
#Removal of stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()
#common word removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()
#Rare word removal
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq
freq = list(freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['tweet'].head()
#Spelling correction
from textblob import TextBlob
train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

#Tokenization
TextBlob(train['tweet'][1]).words
#Stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
#Lemmatization
from textblob import Word
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()
#N-Grams
TextBlob(train['tweet'][0]).ngrams(2)

#Term Frequency
#TF = (Number of times term T appears in the particular row) / (number of terms in that row)
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1
#Inverse Document Frequency
for i,words in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(words)])))
tf1
#TF-IDF
tf1['tfidf'] = tf1['tf']*tf1['idf']
tf1
#tfidf using sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words='english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])

train_vect
#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = 'word')
train_bow = bow.fit_transform(train['tweet'])
train_bow
#Sentiment Analysis
#sentiment as value near to 1 mean positive setiment
#value near to -1 maen negative sentiment
train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0])
train[['tweet','sentiment']].head()
