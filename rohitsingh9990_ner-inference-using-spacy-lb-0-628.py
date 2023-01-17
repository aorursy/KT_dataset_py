import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# !pip install chart_studio



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/ import string

from pandas_profiling import ProfileReport

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec 

from gensim.models import KeyedVectors 

import matplotlib.pyplot as plt

import pickle

from tqdm import tqdm

import os

import nltk

import seaborn as sns

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from plotly import tools

# import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch



from collections import Counter # suppress warnings

import warnings

warnings.filterwarnings("ignore")

sns.set(style="ticks", color_codes=True)

BASE_PATH = '../input/tweet-sentiment-extraction/'

MODELS_BASE_PATH = '../input/tse-spacy-model/models/'

MODELS_BASE_PATH2 = '../input/tse-spacy-model/models2/'



test_df = pd.read_csv( BASE_PATH + 'test.csv')

submission_df = pd.read_csv( BASE_PATH + 'sample_submission.csv')
def predict_entities(text, model):

    doc = model(text)

    ent_array = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        if new_int not in ent_array:

            ent_array.append([start, end, ent.label_])

    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text

    return selected_text
selected_texts = []



if MODELS_BASE_PATH is not None:

    print("Loading Models  from ", MODELS_BASE_PATH)

    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')

    model_neg = spacy.load(MODELS_BASE_PATH2 + 'model_neg')

    model_neu = spacy.load(MODELS_BASE_PATH + 'model_neu')

        

    for index, row in test_df.iterrows():

        text = row.text

        output_str = ""

        if row.sentiment == 'neutral' or len(text.split()) < 4:

#             output_str = text

#             selected_texts.append(predict_entities(text, model_neu))

            selected_texts.append(text)

        elif row.sentiment == 'positive':

            selected_texts.append(predict_entities(text, model_pos))

        else:

            selected_texts.append(predict_entities(text, model_neg))

        

test_df['selected_text'] = selected_texts
test_df.head(10)
submission_df['selected_text'] = test_df['selected_text']

submission_df.to_csv("submission.csv", index=False)

display(submission_df.head(10))