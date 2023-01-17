import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def count_ngrams(dataframe,column,begin_ngram,end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(df['title'].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df,column,number_of_words):

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



def word_bar_graph_function(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()

    

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')  

three_gram = count_ngrams(df,'title',3,3)

words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
plt.figure(figsize=(10,10))

word_bar_graph_function(df,'title','Most common words in the titles of the papers in the CORD-19 dataset')
fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Most Common 3-Words in Titles of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
value_counts = df['journal'].value_counts()

value_counts_df = pd.DataFrame(value_counts)

value_counts_df['journal_name'] = value_counts_df.index

value_counts_df['count'] = value_counts_df['journal']

fig = px.bar(value_counts_df[0:20], 

             x="count", 

             y="journal_name",

             title='Most Common Journals in the CORD-19 Dataset',

             orientation='h')

fig.show()
value_counts = df['publish_time'].value_counts()

value_counts_df = pd.DataFrame(value_counts)

value_counts_df['which_year'] = value_counts_df.index

value_counts_df['count'] = value_counts_df['publish_time']

fig = px.bar(value_counts_df[0:5], 

             x="count", 

             y="which_year",

             title='Most Common Dates of Publication',

             orientation='h')

fig.show()
import json

file_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/252878458973ebf8c4a149447b2887f0e553e7b5.json'

with open(file_path) as json_file:

     json_file = json.load(json_file)

json_file
plt.figure(figsize=(10,10))

word_cloud_function(df,'title',50000)