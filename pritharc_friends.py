Image("../input/giffyy/giphy.gif")
from IPython.display import Image
import os
!ls ../input/
import pandas as pd
import numpy as np
from pandas import DataFrame
train = pd.read_csv('../input/friends-transcript/friends_quotes.csv')
train
train['word_count'] = train['quote'].apply(lambda x: len(str(x).split(" ")))
train[['quote','word_count']].head()
train['char_count'] = train['quote'].str.len() ## this also includes spaces
train[['quote','char_count']].head()
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

train['avg_word'] = train['episode_title'].apply(lambda x: avg_word(x))
train[['quote','avg_word']].head()
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['quote'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['quote','stopwords']].head()
train['hastags'] = train['quote'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['quote','hastags']].head()
train['numerics'] = train['quote'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['quote','numerics']].head()
train['upper'] = train['quote'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['quote','upper']].head()
train['quote'] = train['quote'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['quote'].head()
train['quote'] = train['quote'].str.replace('[^\w\s]','')
train['quote'].head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['quote'] = train['quote'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['quote'].head()
freq = pd.Series(' '.join(train['quote']).split()).value_counts()[:10]
freq
freq = list(freq.index)
train['quote'] = train['quote'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['quote'].head()
freq = pd.Series(' '.join(train['quote']).split()).value_counts()[-10:]
freq
freq = list(freq.index)
train['quote'] = train['quote'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
train['quote'].head()
from textblob import TextBlob
train['quote'][:5].apply(lambda x: str(TextBlob(x).correct()))
TextBlob(train['quote'][1]).words
from nltk.stem import PorterStemmer
st = PorterStemmer()
train['quote'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
from textblob import Word
train['quote'] = train['quote'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['quote'].head()
TextBlob(train['quote'][0]).ngrams(3)
tf1 = (train['quote'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['quote'].str.contains(word)])))

tf1
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['quote'])

train_vect
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['quote'])
train_bow
train['quote'][:5].apply(lambda x: TextBlob(x).sentiment)
train['sentiment'] = train['quote'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['quote','sentiment']].head()
import matplotlib.pyplot as plt
Sentiment_count=train.groupby('sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['quote'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()
#WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
plt.rcParams['font.size']= 15              
plt.rcParams['savefig.dpi']= 100         
plt.rcParams['figure.subplot.bottom']= .1
plt.figure(figsize=(15,15))
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_words=2000, max_font_size=80,
                      random_state=420).generate(str(train['quote']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("Friends [WordCloud]")
plt.axis('off')
plt.show()
plt.figure(figsize=(15,15))
stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_words=2000, max_font_size=100,
                      random_state=420).generate(str(train['quote']))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.title("Friends <3 [WordCloud]")
plt.axis('off')
plt.show()
