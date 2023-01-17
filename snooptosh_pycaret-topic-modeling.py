!pip install pycaret

!python -m spacy download en_core_web_sm

!python -m textblob.download_corpora

import os

import pandas as pd

import pycaret

from pycaret.nlp import *
df_tweets = pd.read_csv("/kaggle/input/clinton-trump-tweets/tweets.csv")

df_tweets.head()
# filtering data for Hillary Clinton tweets

df_tweets_hc = df_tweets[df_tweets['handle'] == "HillaryClinton"].reset_index(drop=True)

print(df_tweets_hc.shape)

df_tweets_hc.head()
df = df_tweets_hc.sample(1000, random_state=493).reset_index(drop=True)

print(df.shape)

df.head()
# initialize the setup

nlp = setup(data = df, target = 'text', session_id = 493, custom_stopwords = [ 'rt', 'https', 'http', 'co', 'amp'])

# create the model

lda = create_model('lda', num_topics = 6, multi_core = True)
# label the data using trained model

df_lda = assign_model(lda)

df_lda.head()
plot_model(lda, plot='topic_distribution')

plot_model(lda, plot='topic_model')

plot_model(lda, plot='wordcloud', topic_num = 'Topic 5')

plot_model(lda, plot='frequency', topic_num = 'Topic 5')

plot_model(lda, plot='bigram', topic_num = 'Topic 5')

plot_model(lda, plot='trigram', topic_num = 'Topic 5')

plot_model(lda, plot='distribution', topic_num = 'Topic 5')

plot_model(lda, plot='sentiment', topic_num = 'Topic 5')