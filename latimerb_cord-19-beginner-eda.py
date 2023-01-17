import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

import json

from sklearn.feature_extraction.text import CountVectorizer

import warnings 

warnings.filterwarnings('ignore')



    

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')



ex_paper_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/02201e4601ab0eb70b6c26480cf2bfeae2625193.json'

with open(ex_paper_path) as f:

     ex_paper = json.load(f)
ex_paper['metadata']['title']
paragraph_num = 1

ex_paper['body_text'][paragraph_num]['text']
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
all_names = df['journal'].values.astype(str)

[j_names, j_nums] = np.unique(all_names[(~pd.isnull(all_names)) & (all_names!='nan')],return_counts=True)



df_j_names = pd.DataFrame(np.concatenate((j_names.astype(str).reshape(-1,1),

                                          j_nums.astype(int).reshape(-1,1)),axis=1),

                          columns=['journal_name','num_articles'])



df_j_names['num_articles'] = df_j_names['num_articles'].astype(int)



df_j_names = df_j_names.sort_values(by='num_articles').reset_index(drop=True)
df_j_names.tail(10)
j_names_show = df_j_names.iloc[-10:,:]

plt.figure(figsize=(10,10))

y_pos = np.arange(j_names_show.shape[0])

plt.barh(y_pos, j_names_show['num_articles'])

 

# Create names on the y-axis

plt.yticks(y_pos, j_names_show['journal_name'])

plt.xlabel('num articles')

plt.ylabel('journal title')

plt.show()
all_years = df['publish_time'].values.astype(str)

all_4_digit = []

for y in all_years:

    if len(y)==4:

        all_4_digit.append(y)

all_4_digit = np.array(all_4_digit)

[years,counts]=np.unique(all_4_digit.astype(int),return_counts=True)
plt.bar(years,counts)