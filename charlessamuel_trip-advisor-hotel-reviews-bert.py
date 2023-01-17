# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip3 install ktrain
import numpy as np

import pandas as pd

import missingno as msno

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import plotly.graph_objects as go

#import plotly.express as px

import matplotlib.pyplot as plt

import spacy

import tensorflow as tf

from wordcloud import WordCloud, STOPWORDS 

import ktrain

from ktrain import text



from collections import Counter

%matplotlib inline
df = pd.read_csv('../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')

df.head()
df.shape
df.isnull().sum().any()
pos = [4, 5]

neg = [1, 2]

neu = [3]



def sentiment(rating):

  if rating in pos:

    return 2

  elif rating in neg:

    return 0

  else:

    return 1  
df['Sentiment'] = df['Rating'].apply(sentiment)

df.head()
fig = go.Figure([go.Bar(x=df.Sentiment.value_counts().index, y=df.Sentiment.value_counts().tolist())])

fig.update_layout(

    title="Values in each Sentiment",

    xaxis_title="Sentiment",

    yaxis_title="Values")

fig.show()
nlp = spacy.load('en')



def normalize(msg):

    

    doc = nlp(msg)

    res = []

    

    for token in doc:

        if(token.is_stop or token.is_punct or token.is_space):

            pass

        else:

            res.append(token.lemma_.lower())

            

    return res
df['Review'] = df['Review'].apply(normalize)

df.head()
words_collection = Counter([item for sublist in df['Review'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_list = [item for sublist in df['Review'] for item in sublist]

word_string = " ".join(word_list)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=60000, 

                      width=1000,

                      height=650

                         ).generate(word_string)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
pos_df = df[df['Sentiment'] == 2]

words_collection = Counter([item for sublist in pos_df['Review'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_list_pos = [item for sublist in pos_df['Review'] for item in sublist]

word_string_pos = " ".join(word_list)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=40000, 

                      width=1000,

                      height=650

                         ).generate(word_string_pos)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
neu_df = df[df['Sentiment'] == 1]

words_collection = Counter([item for sublist in neu_df['Review'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_list_neu = [item for sublist in neu_df['Review'] for item in sublist]

word_string_neu = " ".join(word_list)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=6000, 

                      width=1000,

                      height=650

                         ).generate(word_string_neu)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
neg_df = df[df['Sentiment'] == 0]

words_collection = Counter([item for sublist in neg_df['Review'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(15))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='PuBuGn', low=0, high=0, axis=0, subset=None)
word_list_neg = [item for sublist in neg_df['Review'] for item in sublist]

word_string_neg = " ".join(word_list)



wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white', 

                      max_words=10000, 

                      width=1000,

                      height=650

                         ).generate(word_string_neg)
plt.figure(figsize=(20,10))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
df['Review'] = df['Review'].apply(lambda m: " ".join(m))

df.head()
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(df, 

                                                                    'Review',

                                                                    label_columns=['Sentiment'],

                                                                    preprocess_mode='bert')
model = text.text_classifier(name='bert',

                             train_data=(x_train, y_train),

                             preproc=preproc)
learner = ktrain.get_learner(model=model,

                             train_data=(x_train, y_train),

                             val_data=(x_test, y_test),

                             batch_size=6)
learner.fit_onecycle(lr=2e-5,

                     epochs=1)