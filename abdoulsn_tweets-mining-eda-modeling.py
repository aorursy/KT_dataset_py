import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 10000)

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,GRU,Dense

from keras.initializers import Constant

import os
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
rawtrain = pd.read_csv('../input/nlp-getting-started/train.csv')

rawtest = pd.read_csv('../input/nlp-getting-started/test.csv')

rawtrain.info()
#lets convert target to object type.

rawtrain['target'] = rawtrain['target'].astype(object)
print('There are {} rows and {} columns in train'.format(rawtrain.shape[0],rawtrain.shape[1]))

print('There are {} rows and {} columns in train'.format(rawtest.shape[0],rawtest.shape[1]))
#NaN values

rawtrain.isnull().sum()  
# Lets make some world Cloud

text = rawtrain.text.values

wordcloud = WordCloud(width = 5000, height = 2500,background_color = 'white',

                      stopwords = STOPWORDS).generate(str(text))



fig = plt.figure(figsize = (16, 10), facecolor = 'k', edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#Keyword countvalues

rawtrain.keyword.value_counts()[:10].plot.bar(color='green',);
rawtrain.groupby('target').target.value_counts().plot.barh()
fig,(ax1,ax2) = plt.subplots(1,2,figsize = (18,6))



tweet_len = rawtrain[rawtrain['target'] == 1]['text'].str.len()

ax1.hist(tweet_len,color = 'black')

ax1.set_title('disaster tweets')





tweet_len = rawtrain[rawtrain['target'] == 0]['text'].str.len()

ax2.hist(tweet_len,color = 'green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (18,5))

tweet_len = rawtrain[rawtrain['target'] == 1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color = 'orange')

ax1.set_title('disaster tweets')



tweet_len = rawtrain[rawtrain['target'] ==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color = 'purple')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,6))

word1 = rawtrain[rawtrain['target'] == 1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word1.map(lambda x: np.mean(x)),ax=ax1,color='red')

ax1.set_title('disaster')



word0 = rawtrain[rawtrain['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word0.map(lambda x: np.mean(x)),ax=ax2,color='green')

ax2.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')
data = pd.concat([rawtrain,  rawtest])

data
data['body_len'] = data['text'].apply(lambda x: len(x) - x.count(" "))

data.head()
import string

def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100



data['punct%'] = data['text'].apply(lambda x: count_punct(x))



data.head()
plt.subplots(1,figsize = (18,8))

bins = np.linspace(0, 200, 40)

plt.hist(data[data['target'] == 1]['body_len'], bins, alpha=0.5, density=True, label='1')

plt.hist(data[data['target'] == 0]['body_len'], bins, alpha=0.5, density=True, label='0')

plt.legend(loc='upper left')

plt.xlim(0,150)

plt.show()
bins = np.linspace(0, 50, 40)

plt.subplots(1,figsize = (18,8))

plt.hist(data[data['target']==1]['punct%'], bins, alpha=0.5, density=True, label='1')

plt.hist(data[data['target']==0]['punct%'], bins, alpha=0.5, density=True, label='0')

plt.legend(loc='upper right')

plt.xlim(0,45)

plt.show()
data[['punct%', 'body_len']].kurt()
for i in range(1,7):

    fig = plt.subplots(figsize=(10,4))

    plt.hist((data['punct%'])**(1/i), bins=35)

    plt.hist((data['body_len'])**(1/i), bins=35)

    plt.title(i)

    plt.show()
data['punct%tr'] = data['punct%']**(1/3)
def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    links = re.compile(r'https?://\S+|www\.\S+')

    text = links.sub(r'',text)

    tags = re.compile(r'<.*?>')

    text = tags.sub(r'',text)

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    text =  emoji_pattern.sub(r'', text)

    tokens = re.split('\W+', text)

    #text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text

data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x.lower()))

data.head(20)
import nltk

ps = nltk.PorterStemmer()

stopword = nltk.corpus.stopwords.words('english')

def stemming(text):

    tokens = re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopword]

    return text

data['stem_text'] = data['cleaned_text'].apply(lambda x: stemming(x))


tfidf_vect = TfidfVectorizer(analyzer=stemming)

X_tfidf = tfidf_vect.fit_transform(data['cleaned_text'])

tfidframe = pd.DataFrame(X_tfidf.toarray())



# For concate

data = data.reset_index(drop=True)

X_features = pd.concat([data['body_len'], data['punct%'], tfidframe], axis=1)

X_features.head()
### Re organize data for algoritms

train = X_features[:rawtrain.shape[0]]

test = X_features[rawtrain.shape[0]:]

y_train = rawtrain['target'].astype('int')
type(y_train.values)


rf = RandomForestClassifier()

param = {'n_estimators': [10, 50, 100],

        'max_depth': [10, 20,  None]}



gs = GridSearchCV(rf, param, cv=5, n_jobs=-1)

gs_fit = gs.fit(train, y_train)

pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]