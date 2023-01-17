! pip install --quiet chart-studio
import numpy as np

import dask as pd

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

import fastai

from fastai import *

from fastai.text import *

import os

import chart_studio.plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.express as px

from plotly.subplots import make_subplots

from plotly.offline import iplot

from wordcloud import WordCloud

from plotly.offline import iplot

import re
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

plt.style.use('fivethirtyeight')
data = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")

data.head()
train_data = data[['airline_sentiment', 'text']]

train_data.head()
train_data.info()

train_data.describe()
train_data.isna().sum()
vals = [len(train_data[train_data['airline_sentiment']=='negative']['airline_sentiment']), len(train_data[train_data['airline_sentiment']=='positive']['airline_sentiment']), len(train_data[train_data['airline_sentiment']=='neutral']['airline_sentiment'])]

idx = ['negative', 'positive', 'neutral']

fig = px.pie(

    train_data,

    names='airline_sentiment',

    title='Target Value Distribution Chart',

    height=400,

    width=750

)

iplot(fig)
neg = train_data[train_data['airline_sentiment']=='negative']['text'].str.len()

pos = train_data[train_data['airline_sentiment']=='positive']['text'].str.len()

neu = train_data[train_data['airline_sentiment']=='neutral']['text'].str.len()



fig = make_subplots(rows=1, cols=3)



fig.add_trace(

    go.Histogram(x=list(neg), name='Negative Tweets'),

    row=1, 

    col=1

)



fig.add_trace(

    go.Histogram(x=list(pos), name='Positive Tweets'),

    row=1, 

    col=2,

)



fig.add_trace(

    go.Histogram(x=list(neu), name='Neutral Tweets'),

    row=1, 

    col=3,

)





fig.update_layout(height=400, width=800, title_text="Character Count")

iplot(fig)
neg = train_data[train_data['airline_sentiment']=='negative']['text'].str.split().map(lambda x: len(x))

pos = train_data[train_data['airline_sentiment']=='positive']['text'].str.split().map(lambda x: len(x))

neu = train_data[train_data['airline_sentiment']=='neutral']['text'].str.split().map(lambda x: len(x))



fig = make_subplots(rows=1, cols=3)



fig.add_trace(

    go.Histogram(x=list(neg), name='Negative Tweets'),

    row=1, 

    col=1

)



fig.add_trace(

    go.Histogram(x=list(pos), name='Positive Tweets'),

    row=1, 

    col=2,

)



fig.add_trace(

    go.Histogram(x=list(neu), name='Neutral Tweets'),

    row=1, 

    col=3,

)



fig.update_layout(height=500, width=850, title_text="Word Count")

iplot(fig)
neg = train_data[train_data['airline_sentiment']=='negative']['text'].str.split().map(lambda x: [len(j) for j in x]).map(lambda x: np.mean(x)).to_list()

pos = train_data[train_data['airline_sentiment']=='positive']['text'].str.split().map(lambda x: [len(j) for j in x]).map(lambda x: np.mean(x)).to_list()

neu = train_data[train_data['airline_sentiment']=='neutral']['text'].str.split().map(lambda x: [len(j) for j in x]).map(lambda x: np.mean(x)).to_list()





fig = ff.create_distplot([neg, pos, neu], ['Negative', 'Positive', 'Neutral'])

fig.update_layout(height=500, width=800, title_text="Average Word Length Distribution")

iplot(fig)
neg = train_data[train_data['airline_sentiment']=='negative']['text'].apply(lambda x: len(set(str(x).split()))).to_list()

pos = train_data[train_data['airline_sentiment']=='positive']['text'].apply(lambda x: len(set(str(x).split()))).to_list()

neu = train_data[train_data['airline_sentiment']=='neutral']['text'].apply(lambda x: len(set(str(x).split()))).to_list()



fig = ff.create_distplot([neg, pos, neu], ['Negative', 'Positive', 'Neutral'])

fig.update_layout(height=500, width=800, title_text="Unique Word Count Distribution")

iplot(fig)
neg = train_data[train_data['airline_sentiment']=='negative']['text'].str.split().map(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'ftp' in w]))

pos = train_data[train_data['airline_sentiment']=='positive']['text'].str.split().map(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'ftp' in w]))

neu = train_data[train_data['airline_sentiment']=='neutral']['text'].str.split().map(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w or 'ftp' in w]))



fig = make_subplots(rows=1, cols=3)



fig.add_trace(

    go.Histogram(x=list(neg), name='Negative Tweets'),

    row=1, 

    col=1

)



fig.add_trace(

    go.Histogram(x=list(pos), name='Positive Tweets'),

    row=1, 

    col=2,

)



fig.add_trace(

    go.Histogram(x=list(neu), name='Neutral Tweets'),

    row=1, 

    col=3,

)



fig.update_layout(height=500, width=850, title_text="URL Count")

iplot(fig)
negative = " ".join(train_data[train_data['airline_sentiment'] == 'negative']['text'].to_list())

positive = " ".join(train_data[train_data['airline_sentiment'] == 'positive']['text'].to_list())

neutral = " ".join(train_data[train_data['airline_sentiment'] == 'neutral']['text'].to_list())



fig, ax = plt.subplots(1, 3, figsize=(15,15))

ng_wlc = WordCloud(width=256, height=256, collocations=False).generate(negative)

ps_wlc = WordCloud(width=256, height=256, collocations=False).generate(positive)

ne_wlc = WordCloud(width=256, height=256, collocations=False).generate(neutral)

wcs = [ng_wlc, ps_wlc, ne_wlc]

titls = ["Negative Tweets", "Positive Tweets", "Neutral Tweets"]



for num, el in enumerate(wcs):

    ax[num].imshow(el)

    ax[num].axis('off')

    ax[num].set_title(titls[num])
# Get the stopwords

st_wrds = stopwords.words("english")
# Remove everything except basic text characters

train_data['text'] = train_data['text'].str.replace("[^a-zA-Z]", " ").str.lower()

train_data.sample(5)
# Tokenizing the data

tokenized_data = train_data['text'].apply(lambda x: x.split())

tokenized_data = tokenized_data.apply(lambda x: [word for word in x if word not in st_wrds])
tokenized_data.sample(5)
# Replace the normal text with tokenized text

tok = []

for i in range(len(train_data)):

    t = ' '.join(tokenized_data[i])

    tok.append(t)

train_data['text'] = tok

train_data.sample(5)
# Change the column name and encode the labels

train_data = train_data.rename(columns={'airline_sentiment':'label'})

train_data['label'] = train_data['label'].apply(lambda x: 0 if x=='negative' else (1 if x=='positive' else 2))
# Let us now split the dataset into training and validation sets

split_pcent = 0.15  # How much percent of data should go into testing set

split = int(split_pcent * len(train_data))



shuffled_set = train_data.sample(frac=1).reset_index(drop=True)   # Shuffle the data

valid_set = shuffled_set[:split]   # Get everything till split number

train_set = shuffled_set[split:]   # Get everything after split number
# Make a Language Model Data Bunch from our train set

data_bunch = TextLMDataBunch.from_df(train_df=train_set, valid_df=valid_set, path="")
# Make the data classifier

data_clf = TextClasDataBunch.from_df(path="", train_df=train_set, valid_df=valid_set, vocab=data_bunch.train_ds.vocab, bs=16)
# Define the language learner model and fit for one epoch

learner = language_model_learner(data_bunch, arch=AWD_LSTM, drop_mult=0.5)



learner.fit_one_cycle(1, 1e-2)
# Try unfreezing last 3 layers first

layers_to_unfreeze = [1, 2, 3]

for i in layers_to_unfreeze:

    learner.freeze_to(-i)

    learner.fit_one_cycle(1, 1e-2)
# Now let's unfreeze all layers and train them

learner.unfreeze()

learner.fit_one_cycle(1, 1e-2)
learner.save_encoder('learn_encoder')
clf = text_classifier_learner(data_clf, arch=AWD_LSTM, drop_mult=0.5)

clf.load_encoder('learn_encoder')
clf.fit_one_cycle(1, 1e-2)
# Let's unfreeze all it's layers and train it.

clf.unfreeze()

clf.fit_one_cycle(1)
# Unfreeze last layer and give it a learning rate range using `slice()` function

# This way it'll use the learning rates from 5e-3/2->5e-3 (i.e: 0.0025 -> 0.005)

clf.freeze_to(-1)

clf.fit_one_cycle(1, slice(5e-3/2., 5e-3))
# No let's unfreeze all the layers and try DFT again

clf.unfreeze()

clf.fit_one_cycle(1, slice(2e-3/100, 2e-3))
# The Classifier classifies it Neutral, which is right

clf.predict("Hello, how are you doing?")
# The Classifier classifier it Negative, which is right

clf.predict("Wow, the flight duration was boring and the passenger treatement was not the best I have seen!")
# The Classifier classifier it Positive, which is right

clf.predict("Great service and good staff, I would recommend it!")