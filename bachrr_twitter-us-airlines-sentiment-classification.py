import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from pathlib import Path

from fastai.text import *

from fastai.widgets import ClassConfusion

from sklearn.model_selection import train_test_split



Path.ls= lambda self: list(self.glob("*"))
path = Path('../input/twitter-airline-sentiment')

path.ls()
df = pd.read_csv(path/"Tweets.csv")

df.head()
df['text'].apply(lambda text: len(text)).hist(figsize=(10, 5))
df.groupby("airline_sentiment")['airline_sentiment'].count().plot(kind='bar', figsize=(10, 6))
arline_by_sentiment = df.groupby(['airline', 'airline_sentiment'])['airline'].agg('count')

arline_by_sentiment.unstack().plot(kind='bar', figsize=(10, 6))
location_by_sentiment = df.groupby(['tweet_location', 'airline_sentiment'])['airline'].agg('count')

location_by_sentiment
log_retweet_by_sentiment = np.log(df.groupby(['retweet_count', 'airline_sentiment'])['airline'].agg('count'))

log_retweet_by_sentiment.unstack().plot(kind='bar', figsize=(20, 10))
train_df, valid_df = train_test_split(df, test_size=0.2)
# Language model data

data_lm = TextLMDataBunch.from_df('.', train_df, valid_df, text_cols='text')
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(1, 1e-3)
learn.predict("This is really a bad thing", n_words=10)
learn.predict("This is really a bad thing", n_words=50)
learn.save_encoder('ft_enc')
data_clas = TextClasDataBunch.from_df('.', train_df, valid_df, text_cols='text', label_cols='airline_sentiment', vocab=data_lm.train_ds.vocab, bs=32)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('ft_enc')
learn.fit_one_cycle(1, 1e-2)
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(5e-3/2., 5e-3))
learn.unfreeze()

learn.fit_one_cycle(1, slice(2e-3/100, 2e-3))
learn.fit_one_cycle(10, slice(2e-3/100, 2e-3))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()