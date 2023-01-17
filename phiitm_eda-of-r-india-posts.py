import numpy as np

import pandas as pd 

import plotly.graph_objects as go

import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer

import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

stopWords = stopwords.words('english')

RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
data = pd.read_csv('../input/reddit-india-flair-detector/rindia_ver2.csv')

data.head()
data['Title Length'] = data['Title'].astype(str).apply(len)

data.head()
data.isnull().sum()
data.dropna(subset=['Title','Date'],inplace=True)
fig = go.Figure(data=[go.Bar(

                x = data['Flair'].value_counts()[:25].index.tolist(),

                y = data['Flair'].value_counts()[:25].values.tolist())])



fig.show()
fig = px.histogram(data, x="Title Length")

fig.show()
data['Date'] = data['Date'].apply(lambda x: x[:10])
fig = px.histogram(data, x="Date")

fig.show()
def clean_text(text):

    text = re.sub('#', '', text)  # remove hashtags

    text = re.sub('@\S+', '', text)  # remove mentions

    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""), '', text)  # remove punctuations

    text = re.sub('\s+', ' ', text)  # remove extra whitespace

    text = RE_EMOJI.sub('',text)

    words = word_tokenize(text)

    clean_text = []

    for word in words:

        if word not in stopWords:

            clean_text.append(word)

    cln_txt = ' '.join(clean_text)

    return cln_txt.lower()
data['Clean Title'] = data['Title'].apply(clean_text)

data.head()
def get_top_n_words(corpus, n=None):

    vec = CountVectorizer(stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

common_words = get_top_n_words(data['Clean Title'], 20)

df1 = pd.DataFrame(common_words, columns = ['TitleText' , 'count'])

unigram = df1.groupby('TitleText').sum()['count'].sort_values(ascending=False)

fig = go.Figure(data=[go.Bar(

                y = unigram.tolist(),

                x = unigram.index.tolist())])



fig.show()
def get_top_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2),stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

common_words = get_top_bigram(data['Clean Title'], 20)

df2 = pd.DataFrame(common_words, columns = ['TitleText' , 'count'])

bigram = df2.groupby('TitleText').sum()['count'].sort_values(ascending=False)

fig2 = go.Figure(data=[go.Bar(

                y = bigram.tolist(),

                x = bigram.index.tolist())])



fig2.show()
def get_top_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3),stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

common_words = get_top_trigram(data['Clean Title'], 20)

df3 = pd.DataFrame(common_words, columns = ['TitleText' , 'count'])

trigram = df3.groupby('TitleText').sum()['count'].sort_values(ascending=False)

fig3 = go.Figure(data=[go.Bar(

                y = trigram.tolist(),

                x = trigram.index.tolist())])



fig3.show()