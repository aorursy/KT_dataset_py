# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
instrument_data = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')

instrument_data.head()
# import modules



import re

import string

import nltk

from nltk import pos_tag

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from wordcloud import WordCloud,STOPWORDS

from nltk.stem import WordNetLemmatizer

import tensorflow as tf

from gensim.models import Word2Vec

from keras.layers import Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Embedding, Activation

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from sklearn.model_selection import train_test_split
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
instrument_data = instrument_data.drop(columns=['reviewerID','asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'])

instrument_data.head()
instrument_data = instrument_data.rename(columns={'overall':'rating'})

instrument_data['text'] = instrument_data['reviewText'] + ' ' + instrument_data['summary']
instrument_data = instrument_data.drop(columns=['reviewText', 'summary'])

instrument_data.head()
instrument_data.rating.value_counts()
# visualize chart of rating



instrument_data['rating'].value_counts().iplot(kind='bar',

                                              xTitle='Rating',

                                              yTitle='Count',

                                              title='Rating frequency')
# define for sentiment



def sentiment_rating(rating):

    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)

    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):

        return 0

    else: 

        return 1
instrument_data['sentiment'] = instrument_data['rating'].apply(lambda x : sentiment_rating(x))

instrument_data.head()
instrument_data['sentiment'].value_counts()
# visualize the sentiment counts



instrument_data['sentiment'].value_counts().iplot(kind='bar',

                                                 xTitle='Sentiment',

                                                 yTitle='Count',

                                                 title='Sentiment frequency')
instrument_data.isnull().sum()

instrument_data.dropna(inplace=True)
# Setting up stopwords

stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)



stop.update(punctuation)
def get_simple_pos(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN

    

    

# Lemmatizing words that are not stopwords



lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            final_text.append(word.lower())

    return " ".join(final_text)
instrument_data['text'] = instrument_data['text'].apply(lambda x : lemmatize_words(x))
instrument_data.head()
from sklearn.feature_extraction.text import CountVectorizer



def top_n_ngram(corpus,n = None,ngram = 1):

    vec = CountVectorizer(stop_words = 'english',ngram_range=(ngram,ngram)).fit(corpus)

    bag_of_words = vec.transform(corpus) #Have the count of  all the words for each review

    sum_words = bag_of_words.sum(axis =0) #Calculates the count of all the word in the whole review

    words_freq = [(word,sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq,key = lambda x:x[1],reverse = True)

    return words_freq[:n]
# Top 20 unigrams



common_words = top_n_ngram(instrument_data['text'], 20,1)

df = pd.DataFrame(common_words, columns = ['text' , 'count'])

df.groupby('text').sum()['count'].sort_values(ascending=False).iplot(

kind='bar', title='Top 20 unigrams in review after removing stop words')
# Top 20 bigrams



common_words = top_n_ngram(instrument_data['text'], 20,2)

df = pd.DataFrame(common_words, columns = ['text' , 'count'])

df.groupby('text').sum()['count'].sort_values(ascending=False).iplot(

kind='bar', title='Top 20 bigrams in review after removing stop words')
# Top 20 trigrams



common_words = top_n_ngram(instrument_data['text'], 20,3)

df = pd.DataFrame(common_words, columns = ['text' , 'count'])

df.groupby('text').sum()['count'].sort_values(ascending=False).iplot(

kind='bar', title='Top 20 trigrams in review after removing stop words')
i_d = []

for i in instrument_data['text']:

    i_d.append(i.split())

print(i_d[:2])
# initiate word2vec model



w2v_model = Word2Vec(i_d, size=20, workers=32, min_count=1, window=3)

print(w2v_model)
# tokenize the data



tokenizer = Tokenizer(50622)

tokenizer.fit_on_texts(instrument_data['text'])

text = tokenizer.texts_to_sequences(instrument_data['text'])

text = pad_sequences(text)
y = instrument_data['sentiment']
# split the data into training and testing data



X_train, X_test, y_train, y_test = train_test_split(np.array(text), y, test_size=0.2, stratify=y)
# train the model



model = Sequential()

model.add(w2v_model.wv.get_keras_embedding(True))

model.add(Dropout(0.2))

model.add(Conv1D(20, 3, activation='relu', padding='same', strides=1))

model.add(MaxPool1D())

model.add(Dropout(0.2))

model.add(Conv1D(40, 3, activation='relu', padding='same', strides=1))

model.add(MaxPool1D())

model.add(Dropout(0.2))

model.add(Conv1D(80, 3, activation='relu', padding='same', strides=1))

model.add(GlobalMaxPool1D())

model.add(Dropout(0.2))

model.add(Dense(80))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')

model.summary()
# train the model



history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))