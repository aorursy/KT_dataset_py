# init

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import time

import warnings 

warnings.filterwarnings('ignore')
# define some functions

def count_ngrams(dataframe, column, begin_ngram, end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(df[column].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df, column, number_of_words):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=number_of_words,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def word_bar_graph_function(df, column, title, nvals=50):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(nvals), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:nvals])])

    plt.yticks([x + 0.5 for x in range(nvals)], reversed(popular_words_nonstop[0:nvals]))

    plt.title(title)

    plt.show()
# load metadata

t1 = time.time()

# df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') # adjust to change in data

t2 = time.time()

print('Elapsed time:', t2-t1)
df.head()
df.describe(include='all')
df.journal.value_counts()
# plot top 10 only

df.journal.value_counts()[0:10].plot(kind='bar')

plt.grid()

plt.show()
df.source_x.value_counts().plot(kind='bar')

plt.show()
df.has_pdf_parse.value_counts().plot(kind='bar')

plt.show()
df.publish_time.value_counts()
df.license.value_counts()
# show example

df.title[0]
# show example

df.title[1]
# show most frequent words in titles

plt.figure(figsize=(10,10))

word_bar_graph_function(df,column='title', 

                        title='Most common words in the TITLES of the papers in the CORD-19 dataset',

                        nvals=20)
# evaluate 3-grams (takes some time)

t1 = time.time()

three_gram = count_ngrams(df,'title',3,3)

t2 = time.time()

print('Elapsed time:', t2-t1)
three_gram[0:20]
# plot most frequent 3-grams

fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Top Ten 3-Grams in TITLES of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
# evaluate bigrams (takes some time)

t1 = time.time()

bi_gram = count_ngrams(df,'title',2,2)

t2 = time.time()

print('Elapsed time:', t2-t1)
bi_gram[0:20]
# plot most frequent bigrams

fig = px.bar(bi_gram.sort_values('frequency',ascending=False)[2:12], 

             x="frequency", 

             y="ngram",

             title='Top Ten relevant bigrams in TITLES of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
# word cloud

plt.figure(figsize=(10,10))

word_cloud_function(df,column='title',number_of_words=50000)
def word_finder(i_word, i_text):

    found = str(i_text).find(i_word)

    if found == -1:

        result = 0

    else:

        result = 1

    return result
# define keyword

my_keyword = 'enzyme'
# partial function for mapping

word_indicator_partial = lambda text: word_finder(my_keyword, text)

# build indicator vector (0/1) of hits

keyword_indicator = np.asarray(list(map(word_indicator_partial, df.title)))
# number of hits

print('Number of hits for keyword <', my_keyword, '> : ', keyword_indicator.sum())
# add index vector as additional column

df['selection'] = keyword_indicator
# select only hits from data frame

df_hits = df[df['selection']==1]
# show results

df_hits
# store result in CSV file

df_hits.to_csv('demo_keyword_search.csv')
# show example

df.abstract[3]
# show most frequent words in abstracts

plt.figure(figsize=(10,10))

word_bar_graph_function(df,column='abstract',

                        title='Most common words in the ABSTRACTS of the papers in the CORD-19 dataset',

                        nvals=20)
# word cloud

plt.figure(figsize=(10,10))

word_cloud_function(df,column='abstract',number_of_words=50000)
# evaluate 3-grams (takes some time)

t1 = time.time()

three_gram_abs = count_ngrams(df,'abstract',3,3)

t2 = time.time()

print('Elapsed time:', t2-t1)
three_gram_abs[0:20]
# plot most frequent 3-grams

fig = px.bar(three_gram_abs.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Top Ten 3-Grams in ABSTRACTS of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
# evaluate bigrams (takes some time)

t1 = time.time()

bi_gram_abs = count_ngrams(df,'abstract',2,2)

t2 = time.time()

print('Elapsed time:', t2-t1)
bi_gram_abs[0:50]