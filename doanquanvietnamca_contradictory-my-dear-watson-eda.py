!pip install -q pyicu

!pip install -q pycld2

!pip install -q polyglot

!pip install -q textstat

!pip install -q googletrans
import warnings

warnings.filterwarnings("ignore")



import os

import gc

import re

import folium

import textstat

from scipy import stats

from colorama import Fore, Back, Style, init



import math

import numpy as np

import scipy as sp

import pandas as pd



import random

import networkx as nx

from pandas import Timestamp



from PIL import Image

from IPython.display import SVG

from keras.utils import model_to_dot



import requests

from IPython.display import HTML



import seaborn as sns

from tqdm import tqdm

import matplotlib.cm as cm

import matplotlib.pyplot as plt



tqdm.pandas()



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

plt.style.use('ggplot')
train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

train.head()
lang_list = sorted(list(set(train["language"])))

counts = [list(train["language"]).count(cont) for cont in lang_list]

df = pd.DataFrame(np.transpose([lang_list, counts]))

df.columns = ["Language", "Count"]

df["Count"] = df["Count"].apply(int)





fig = px.bar(df, x="Language", y="Count", title="Language of train data", color="Language", text="Count")

fig.update_layout(template="plotly_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig.data[0].textfont.color = "black"

fig.data[0].textposition = "outside"

fig.data[1].textfont.color = "black"

fig.data[1].textposition = "outside"

fig
def get_country(language):

    if language == "German":

        return "Germany"

    if language == "Bulgarian":

        return "Bulgary"

    if language == "Chinese":

        return "China"

    if language == "Arabic":

        return "Saudi Arabia"

    if language == "Hindi":

        return "India"

    if language == "Thai":

        return "Thailand"

    if language == "Urdu":

        return "Pakistan"

    if language == "Swahili":

        return "Tanzania"

    if language == "English":

        return "United Kingdom"

    if language == "Hindi":

        return "India"

    if language == "French":

        return "France"

    if language == "Greek":

        return "Greece"

    if language == "Spanish":

        return "Spain"

    if language == "Russian":

        return "Russia"

    if language == "Vietnamese":

        return "Vietnam"

    return "None"

    

df["country"] = df["Language"].progress_apply(get_country)
fig = px.choropleth(df.query("Language != 'English' and Language != 'un' and country != 'None'").query("Count >= 5"),

                    locations="country", hover_name="country",

                     projection="natural earth", locationmode="country names", title="Countries of train",

                    color="Count", template="plotly", color_continuous_scale="agsunset")

# fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

# fig.data[0].marker.line.width = 0.2

fig.show()
def new_len(x):

    if type(x) is str:

        return len(x.split())

    else:

        return 0



train["premise"] = train["premise"].apply(new_len)

nums = train.query("premise != 0 and premise < 200").sample(frac=0.1)["premise"]

fig = ff.create_distplot(hist_data=[nums],

                         group_labels=["All premise"],

                         colors=["coral"])



fig.update_layout(title_text="Premise words", xaxis_title="Premise words", template="simple_white", showlegend=False)

fig.show()
train["hypothesis"] = train["hypothesis"].apply(new_len)

nums = train.query("hypothesis != 0 and hypothesis < 200").sample(frac=0.1)["hypothesis"]

fig = ff.create_distplot(hist_data=[nums],

                         group_labels=["All hypothesis"],

                         colors=["coral"])



fig.update_layout(title_text="Hypothesis words", xaxis_title="hypothesis words", template="simple_white", showlegend=False)

fig.show()
df = pd.DataFrame(np.transpose([lang_list, train.groupby("language").mean()["hypothesis"]]))

df.columns = ["Language", "Average_comment_words"]

df["Average_comment_words"] = df["Average_comment_words"].apply(float)

df = df.query("Average_comment_words < 500")

fig = go.Figure(go.Bar(x=df["Language"], y=df["Average_comment_words"]))



fig.update_layout(xaxis_title="Language", yaxis_title="Average hypothesis words", title_text="Average hypothesis words vs. language", template="plotly_white")

fig.show()
df = pd.DataFrame(np.transpose([lang_list, train.groupby("language").mean()["premise"]]))

df.columns = ["Language", "Average_comment_words"]

df["Average_comment_words"] = df["Average_comment_words"].apply(float)

df = df.query("Average_comment_words < 500")

fig = go.Figure(go.Bar(x=df["Language"], y=df["Average_comment_words"]))



fig.update_layout(xaxis_title="Language", yaxis_title="Average premise words", title_text="Average premise words vs. language", template="plotly_white")

fig.show()
fig = go.Figure(data=[

    go.Pie(labels=train['label'].value_counts().index,

           values=train['label'].value_counts().values, marker=dict(colors=px.colors.qualitative.Plotly))

])

fig.update_traces(textposition='outside', textfont=dict(color="black"))

fig.update_layout(title_text="Pie chart of labels")

fig.show()
per_lang = train.groupby(by=['language', 'label']).count()[['id']]



data=[]

for lang in train['language'].unique():

      y = per_lang[per_lang.index.get_level_values('language') == lang].values.flatten()

      data.append(go.Bar(name=lang, x=['entailment', 'contradiction', 'neutral'], y=y))

fig = go.Figure(data=data)

fig.update_layout(

    title='Language distribution in the train dataset',

    barmode='group'

)

fig.show()
from wordcloud import WordCloud, STOPWORDS

train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')



rnd_comments = train[train['label'] == 0]['hypothesis'].values

wc = WordCloud(background_color="black", max_words=2000)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in premise", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
from wordcloud import WordCloud, STOPWORDS



rnd_comments = train[train['label'] == 0]['premise'].values

wc = WordCloud(background_color="black", max_words=2000)

wc.generate(" ".join(rnd_comments))



plt.figure(figsize=(20,10))

plt.axis("off")

plt.title("Frequent words in premise", fontsize=20)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

# Define helper functions

def get_top_n_words(n_top_words, count_vectorizer, text_data):

    '''

    returns a tuple of the top n words in a sample and their 

    accompanying counts, given a CountVectorizer object and text sample

    '''

    vectorized_headlines = count_vectorizer.fit_transform(train.hypothesis.values)

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)

    word_values = np.flip(np.sort(vectorized_total)[0,:],1)

    

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))

    for i in range(n_top_words):

        word_vectors[i,word_indices[0,i]] = 1



    words = [word[0].encode('ascii',errors="ignore").decode('utf-8',errors="ignore") for 

             word in count_vectorizer.inverse_transform(word_vectors)]



    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=train.hypothesis.values)



fig, ax = plt.subplots(figsize=(10,4))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()