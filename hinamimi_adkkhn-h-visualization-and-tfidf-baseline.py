from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

from datetime import datetime

from pytz import timezone

datetime.now(timezone('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')



def refer_args(x):

    if type(x) == 'method':

        print(*x.__code__.co_varnames.split(), sep='\n')

    else:

        print(*[x for x in dir(x) if not x.startswith('__')], sep='\n')
from collections import Counter, defaultdict

from itertools import chain

import os

import re

import string

import warnings

warnings.simplefilter('ignore')



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

plt.style.use('ggplot')

import seaborn as sns



from nltk.util import ngrams

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

import gensim

from wordcloud import WordCloud



from tqdm import tqdm

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.metrics import classification_report,confusion_matrix

import torch
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")



print(f'There are {df_train.shape[0]} rows and {df_train.shape[0]} columns in train')

print(f'There are {df_test.shape[0]} rows and {df_test.shape[1]} columns in train')

df_train.head(10)
disaster_tweet = df_train[df_train['target']==1]

not_disaster_tweet = df_train[df_train['target']==0]
plt.rcParams['figure.figsize'] = 14.0, 5.0

plt.rcParams['font.size'] = 16



fig, (ax1, ax2) = plt.subplots(1, 2)

_ = ax1.pie(

    [len(disaster_tweet), len(not_disaster_tweet)],

    labels=('Disaster', 'Not disaster'),

    autopct="%1.1f%%"

)

_ = sns.countplot(x='target', data=df_train, axes=ax2)
plt.rcParams['figure.figsize'] = 20.0, 8.0

plt.rcParams['font.size'] = 10



fig, (ax1, ax2) = plt.subplots(1, 2)

corpus_disaster = list(chain(*disaster_tweet['text'].str.split()))

wordcloud = WordCloud(background_color='black', stopwords=stop).generate(' '.join(corpus_disaster))

_ = ax1.imshow(wordcloud)

ax1.set_axis_off()

not_corpus_disaster = list(chain(*not_disaster_tweet['text'].str.split()))

wordcloud = WordCloud(background_color='white', stopwords=stop).generate(' '.join(not_corpus_disaster))

_ = ax2.imshow(wordcloud)

ax2.set_axis_off()
def clean(tweet):

    tweet = re.sub(r"\x89Û_", "", tweet)

    tweet = re.sub(r"\x89ÛÒ", "", tweet)

    tweet = re.sub(r"\x89ÛÓ", "", tweet)

    tweet = re.sub(r"\x89ÛÏWhen", "When", tweet)

    tweet = re.sub(r"\x89ÛÏ", "", tweet)

    tweet = re.sub(r"China\x89Ûªs", "China's", tweet)

    tweet = re.sub(r"let\x89Ûªs", "let's", tweet)

    tweet = re.sub(r"\x89Û÷", "", tweet)

    tweet = re.sub(r"\x89Ûª", "", tweet)

    tweet = re.sub(r"\x89Û\x9d", "", tweet)

    tweet = re.sub(r"å_", "", tweet)

    tweet = re.sub(r"\x89Û¢", "", tweet)

    tweet = re.sub(r"\x89Û¢åÊ", "", tweet)

    tweet = re.sub(r"fromåÊwounds", "from wounds", tweet)

    tweet = re.sub(r"åÊ", "", tweet)

    tweet = re.sub(r"åÈ", "", tweet)

    tweet = re.sub(r"JapÌ_n", "Japan", tweet)    

    tweet = re.sub(r"Ì©", "e", tweet)

    tweet = re.sub(r"å¨", "", tweet)

    tweet = re.sub(r"SuruÌ¤", "Suruc", tweet)

    tweet = re.sub(r"åÇ", "", tweet)

    tweet = re.sub(r"å£3million", "3 million", tweet)

    tweet = re.sub(r"åÀ", "", tweet)

    tweet = re.sub(r"\x89Ûª", "'", tweet)

    

    tweet = re.sub(r'&gt;', '>', tweet)

    tweet = re.sub(r'&lt;', '<', tweet)

    tweet = re.sub(r'&amp;', '&', tweet)

    

    tweet = re.sub(r'https?:\/\/t.co\/[A-Za-z0-9]+', '', tweet)

    

    tweet = re.sub(r'!+', '!', tweet)

    

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    for p in punctuations:

        tweet = tweet.replace(p, f' {p} ')

    return tweet



df_train['cleaned_text'] = df_train['text'].apply(clean)

df_test['cleaned_text'] = df_test['text'].apply(clean)

disaster_tweet['cleaned_text'] = disaster_tweet['text'].apply(clean)

not_disaster_tweet['cleaned_text'] = not_disaster_tweet['text'].apply(clean)
plt.rcParams['figure.figsize'] = 20.0, 8.0

plt.rcParams['font.size'] = 10



fig, (ax1, ax2) = plt.subplots(1, 2)

corpus_disaster = list(chain(*disaster_tweet['cleaned_text'].str.split()))

wordcloud = WordCloud(background_color='black', stopwords=stop).generate(' '.join(corpus_disaster))

_ = ax1.imshow(wordcloud)

ax1.set_axis_off()

not_corpus_disaster = list(chain(*not_disaster_tweet['cleaned_text'].str.split()))

wordcloud = WordCloud(background_color='white', stopwords=stop).generate(' '.join(not_corpus_disaster))

_ = ax2.imshow(wordcloud)

ax2.set_axis_off()
# nb_word: a number of words in text

disaster_tweet['word_count'] = disaster_tweet['text'].apply(lambda x:len(x.split()))

not_disaster_tweet['word_count'] = not_disaster_tweet['text'].apply(lambda x:len(x.split()))

# hashtags

disaster_tweet['hashtags'] = disaster_tweet['text'].apply(

        lambda x:[y for y in x.split() if '#' in y] if '#' in x else pd.NA)

not_disaster_tweet['hashtags'] = not_disaster_tweet['text'].apply(

        lambda x:[y for y in x.split() if '#' in y] if '#' in x else pd.NA)



# Counter(chain(*disaster_tweet['hashtags'].dropna())).most_common(100)

# Counter(chain(*not_disaster_tweet['hashtags'].dropna())).most_common(100)
# count_vectorizer = CountVectorizer()

# cv_train = count_vectorizer.fit_transform(df_train['cleaned_text'])

# cv_test = count_vectorizer.transform(df_test['cleaned_text'])



tfidf_vectorizer = TfidfVectorizer()

tv_train = tfidf_vectorizer.fit_transform(df_train['cleaned_text'])

tv_test = tfidf_vectorizer.transform(df_test['cleaned_text'])



X_train, X_val, y_train, y_val = train_test_split(cv_train, df_train['target'].values, test_size=0.2)

print('Shape of train', X_train.shape)

print('Shape of Validation ', X_val.shape)
# model = Sequential()

# model.add(SpatialDropout1D(0.2))

# model.add(Dense(1000, activation='relu'))

# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

# model.add(Dense(1, activation='sigmoid'))

# optimzer = Adam(learning_rate=3e-4)



# model.compile(loss='binary_crossentropy', optimizer=optimzer, metrics=['accuracy'])



# history = model.fit(

#     X_train,

#     y_train,

#     batch_size=16,

#     epochs=10,

#     validation_data=(X_val, y_val),

#     verbose=2

# )