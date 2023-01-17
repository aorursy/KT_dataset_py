!pip install chart_studio
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import re

import gc

import sys

import time

import warnings

warnings.filterwarnings("ignore")

import emoji

import random

import unicodedata

import multiprocessing

import seaborn as sns

from functools import partial, lru_cache

from tqdm import tqdm_notebook

from wordcloud import WordCloud, STOPWORDS

import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from plotly import tools

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from nltk import TweetTokenizer

from nltk.stem import PorterStemmer, SnowballStemmer

from nltk.stem.lancaster import LancasterStemmer
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train_len, test_len = len(train.index), len(test.index)

print("Train Size:", train_len)

print("Test Size:", test_len)

print("Train shape: ", train.shape)
train.head()
test.head()
train.info()
test.info()
sns.countplot(train["sentiment"])
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))

train['select_num_words'] = train["selected_text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))

train['select_num_unique_words'] = train["selected_text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))

train['select_num_chars'] = train["selected_text"].apply(lambda x: len(str(x)))
def target_plot(column, title):

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=train[str(column)],name = title + ' : train data'))

    fig.add_trace(go.Histogram(x=test[str(column)],name = title + ' : test data'))

    fig.add_trace(go.Histogram(x=train['select_'+str(column)],name =  title + ': selected text'))

    fig.update_layout(barmode='stack')

    fig.update_traces(opacity=0.75)

    fig.show()
target_plot(column='num_words', title="Number of words")
target_plot(column="num_chars", title="Number of characters")
target_plot(column="num_unique_words", title="Number of unique words")
stopwords = set(STOPWORDS)

def generate_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='black',

        stopwords=stopwords,

        max_words=100,

        max_font_size=40, 

        scale=5,

        random_state=23

    ).generate(str(data))



    fig = plt.figure(1, figsize=(20,20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
generate_wordcloud(train['text'], title = 'Prevalent words in tweets - train data')
generate_wordcloud(test['text'], title = 'Prevalent words in tweets - test data')
generate_wordcloud(train.loc[train['sentiment']=='neutral']['text'], title = 'Prevalent Neutral words in tweets - train data')
generate_wordcloud(train.loc[train['sentiment']=='positive']['text'], title = 'Prevalent Positive words in tweets - train data')
generate_wordcloud(train.loc[train['sentiment']=='negative']['text'], title = 'Prevalent Negative words in tweets - train data')
def extract_emojis(string):

    return [c for c in str(string) if c in emoji.UNICODE_EMOJI]



train['emojis_count'] = train['text'].apply(lambda x: len(extract_emojis(x)))

print("Maximum Number of Emojis: ", max(train["emojis_count"]))
PORTER_STEMMER = PorterStemmer()

LANCASTER_STEMMER = LancasterStemmer()

SNOWBALL_STEMMER = SnowballStemmer("english")



def word_forms(word):

    yield word

    yield word.lower()

    yield word.upper()

    yield word.capitalize()

    yield PORTER_STEMMER.stem(word)

    yield LANCASTER_STEMMER.stem(word)

    yield SNOWBALL_STEMMER.stem(word)
TABLE = str.maketrans(

    {

        "\xad": None,

        "\x7f": None,

        "\ufeff": None,

        "\u200b": None,

        "\u200e": None,

        "\u202a": None,

        "\u202c": None,

        "‘": "'",

        "’": "'",

        "`": "'",

        "“": '"',

        "”": '"',

        "«": '"',

        "»": '"',

        "ɢ": "G",

        "ɪ": "I",

        "ɴ": "N",

        "ʀ": "R",

        "ʏ": "Y",

        "ʙ": "B",

        "ʜ": "H",

        "ʟ": "L",

        "ғ": "F",

        "ᴀ": "A",

        "ᴄ": "C",

        "ᴅ": "D",

        "ᴇ": "E",

        "ᴊ": "J",

        "ᴋ": "K",

        "ᴍ": "M",

        "Μ": "M",

        "ᴏ": "O",

        "ᴘ": "P",

        "ᴛ": "T",

        "ᴜ": "U",

        "ᴡ": "W",

        "ᴠ": "V",

        "ĸ": "K",

        "в": "B",

        "м": "M",

        "н": "H",

        "т": "T",

        "ѕ": "S",

        "—": "-",

        "–": "-",

    }

)



RE_SPACE = re.compile(r"\s")

RE_MULTI_SPACE = re.compile(r"\s+")

RE_URL = re.compile('http[s]?://\S+')
def normalize(text: str):

    text = RE_URL.sub("URL", text)

    text = RE_SPACE.sub(" ", text)

    text = unicodedata.normalize("NFKD", text)

    text = text.translate(TABLE)

    text = RE_MULTI_SPACE.sub(" ", text).strip()

    return text
X_train = train.copy()

X_train["text"] = [str(x) for x in X_train["text"]]

X_train["selected_text"] = [str(x) for x in X_train["selected_text"]]
%%time

with multiprocessing.Pool(processes=3) as pool:

    selected_train_list = pool.map(normalize, X_train.selected_text.tolist())

    train_list = pool.map(normalize, X_train.text.tolist())

    test_list = pool.map(normalize, test.text.tolist())
X_train['preprocessed_selected_text'] = selected_train_list

X_train['preprocessed_text'] = train_list

test['preprocessed_text'] = test_list
import pickle

with open('train.pickle', 'wb') as file:

    pickle.dump(X_train, file, protocol=pickle.HIGHEST_PROTOCOL)

with open('test.pickle', 'wb') as file:

    pickle.dump(test, file, protocol=pickle.HIGHEST_PROTOCOL)