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



import transformers

import tensorflow as tf



from tensorflow.keras.callbacks import Callback

from sklearn.metrics import accuracy_score, roc_auc_score

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger



from tensorflow.keras.models import Model

from kaggle_datasets import KaggleDatasets

from tensorflow.keras.optimizers import Adam

from tokenizers import BertWordPieceTokenizer

from tensorflow.keras.layers import Dense, Input, Dropout, Embedding

from tensorflow.keras.layers import LSTM, GRU, Conv1D, SpatialDropout1D



from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras import activations

from tensorflow.keras import constraints

from tensorflow.keras import initializers

from tensorflow.keras import regularizers



import tensorflow.keras.backend as K

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.activations import *

from tensorflow.keras.constraints import *

from tensorflow.keras.initializers import *

from tensorflow.keras.regularizers import *



from sklearn import metrics

from sklearn.utils import shuffle

from gensim.models import Word2Vec

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer



from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer  



import nltk

from textblob import TextBlob



from nltk.corpus import wordnet

from nltk.corpus import stopwords

from googletrans import Translator

from nltk import WordNetLemmatizer

from polyglot.detect import Detector

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS

from nltk.sentiment.vader import SentimentIntensityAnalyzer



stopword=set(STOPWORDS)



lem = WordNetLemmatizer()

tokenizer=TweetTokenizer()



np.random.seed(0)
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

os.listdir(DATA_PATH)
train_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_data.head()
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in train_data["comment_text"]])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Common words in comments')
def new_len(x):

    if type(x) is str:

        return len(x.split())

    else:

        return 0



train_data["comment_words"] = train_data["comment_text"].apply(new_len)

nums = train_data.query("comment_words != 0 and comment_words < 200").sample(frac=0.1)["comment_words"]

fig = ff.create_distplot(hist_data=[nums],

                         group_labels=["All comments"],

                         colors=["coral"])



fig.update_layout(title_text="Comment words", xaxis_title="Comment words", template="simple_white", showlegend=False)

fig.show()
def polarity(x):

    if type(x) == str:

        return SIA.polarity_scores(x)

    else:

        return 1000

    

SIA = SentimentIntensityAnalyzer()

train_data["polarity"] = train_data["comment_text"].progress_apply(polarity)
fig = go.Figure(go.Histogram(x=[pols["neg"] for pols in train_data["polarity"] if pols["neg"] != 0], marker=dict(

            color='seagreen')

    ))



fig.update_layout(xaxis_title="Negativity sentiment", title_text="Negativity sentiment", template="simple_white")

fig.show()
train_data["negativity"] = train_data["polarity"].apply(lambda x: x["neg"])



nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["negativity"]

nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["negativity"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-toxic"],

                         colors=["darkorange", "dodgerblue"], show_hist=False)



fig.update_layout(title_text="Negativity vs. Toxicity", xaxis_title="Negativity", template="simple_white")

fig.show()
fig = go.Figure(go.Histogram(x=[pols["pos"] for pols in train_data["polarity"] if pols["pos"] != 0], marker=dict(

            color='indianred')

    ))



fig.update_layout(xaxis_title="Positivity sentiment", title_text="Positivity sentiment", template="simple_white")

fig.show()
train_data["positivity"] = train_data["polarity"].apply(lambda x: x["pos"])



nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["positivity"]

nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["positivity"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-toxic"],

                         colors=["darkorange", "dodgerblue"], show_hist=False)



fig.update_layout(title_text="Positivity vs. Toxicity", xaxis_title="Positivity", template="simple_white")

fig.show()
fig = go.Figure(go.Histogram(x=[pols["neu"] for pols in train_data["polarity"] if pols["neu"] != 1], marker=dict(

            color='dodgerblue')

    ))



fig.update_layout(xaxis_title="Neutrality sentiment", title_text="Neutrality sentiment", template="simple_white")

fig.show()
train_data["neutrality"] = train_data["polarity"].apply(lambda x: x["neu"])



nums_1 = train_data.sample(frac=0.1).query("toxic == 1")["neutrality"]

nums_2 = train_data.sample(frac=0.1).query("toxic == 0")["neutrality"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-toxic"],

                         colors=["darkorange", "dodgerblue"], show_hist=False)



fig.update_layout(title_text="Neutrality vs. Toxicity", xaxis_title="Neutrality", template="simple_white")

fig.show()