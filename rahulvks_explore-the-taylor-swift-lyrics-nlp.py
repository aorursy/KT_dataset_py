# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import os

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

width = 0.75

from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import stopwords

from collections import defaultdict

import string

from sklearn.preprocessing import StandardScaler

import seaborn as sns

sns.set_palette(sns.color_palette('tab20', 20))

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from datetime import date, timedelta

import operator 

import re

import spacy

from spacy import displacy

from spacy.util import minibatch, compounding

import spacy #load spacy

nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

#stops = stopwords.words("english")

from tqdm import  tqdm

from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

from IPython.display import IFrame

from IPython.core.display import display, HTML



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/taylor_swift_lyrics.csv",encoding = "latin1")

data.head()
def get_features(df):    

    data['lyric'] = data['lyric'].apply(lambda x:str(x))

    data['total_length'] = data['lyric'].apply(len)

    data['capitals'] = data['lyric'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    data['caps_vs_length'] = data.apply(lambda row: float(row['capitals'])/float(row['total_length']),

                                axis=1)

    data['num_words'] = data.lyric.str.count('\S+')

    data['num_unique_words'] = data['lyric'].apply(lambda comment: len(set(w for w in comment.split())))

    data['words_vs_unique'] = data['num_unique_words'] / df['num_words']  

    return df
sns.set(rc={'figure.figsize':(11.7,8.27)})

y1 = data[data['year'] == 2017]['lyric'].str.len()

sns.distplot(y1, label='2017')

y2 = data[data['year'] == 2014]['lyric'].str.len()

sns.distplot(y2, label='2014')

y3 = data[data['year'] == 2012]['lyric'].str.len()

sns.distplot(y3, label='2012')

y4 = data[data['year'] == 2010]['lyric'].str.len()

sns.distplot(y4, label='2010')

y5 = data[data['year'] == 2008]['lyric'].str.len()

sns.distplot(y5, label='2008')

y6 = data[data['year'] == 2006]['lyric'].str.len()

sns.distplot(y6, label='2006')

plt.title('Year Wise - Lyrics Lenght Distribution (Without Preprocessing)')

plt.legend();

train = get_features(data)

data_pair = data.filter(['year','total_length','capitals','caps_vs_length','num_words','num_unique_words','words_vs_unique'],axis=1)
data.head().T
sns.pairplot(data_pair,hue='year',palette="husl");
contraction_mapping_1 = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 

                       "could've": "could have", "couldn't": "could not", "didn't": "did not",  

                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 

                       "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 

                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",

                       "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 

                       "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 

                       "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 

                       "mayn't": "may not", "might've": "might have","mightn't": "might not",

                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 

                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 

                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 

                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 

                       "she'll've": "she will have", "she's": "she is", "should've": "should have", 

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",

                       "so's": "so as", "this's": "this is","that'd": "that would", 

                       "that'd've": "that would have", "that's": "that is", "there'd": "there would", 

                       "there'd've": "there would have", "there's": "there is", "here's": "here is",

                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", "they've": "they have", 

                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 

                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 

                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is",

                       "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", 

                       "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", 

                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 

                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 

                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",

                       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 

                       "you'll've": "you will have", "you're": "you are", "you've": "you have" ,

                       "Isn't":"is not", "\u200b":"", "It's": "it is","I'm": "I am","don't":"do not","did't":"did not","ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 

                       "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 

                       "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 

                       "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 

                       "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 

                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 

                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 

                       "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", 

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 

                       "there'd": "there would", "there'd've": "there would have", "there's": "there is",

                       "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 

                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",

                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 

                       "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",

                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", 

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
def get_features(df):    

    data['Clean_Lyrics'] = data['Clean_Lyrics'].apply(lambda x:str(x))

    data['total_length'] = data['Clean_Lyrics'].apply(len)

    data['capitals'] = data['Clean_Lyrics'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    data['caps_vs_length'] = data.apply(lambda row: float(row['capitals'])/float(row['total_length']),

                                axis=1)

    data['num_words'] = data.lyric.str.count('\S+')

    data['num_unique_words'] = data['Clean_Lyrics'].apply(lambda comment: len(set(w for w in comment.split())))

    data['words_vs_unique'] = data['num_unique_words'] / df['num_words']  

    return df
data['Clean_Lyrics'] = data['lyric'].apply(lambda x: clean_contractions(x, contraction_mapping_1))

#Stopwords

data['Clean_Lyrics'] = data['Clean_Lyrics'].apply(lambda x: ' '.join([word for word in x.split() if word not in (STOPWORDS)]))

#Re-calculate the features

train = get_features(data)
data.head().T
sns.set(rc={'figure.figsize':(11.7,8.27)})

y1 = data[data['year'] == 2017]['Clean_Lyrics'].str.len()

sns.distplot(y1, label='2017')

y2 = data[data['year'] == 2014]['Clean_Lyrics'].str.len()

sns.distplot(y2, label='2014')

y3 = data[data['year'] == 2012]['Clean_Lyrics'].str.len()

sns.distplot(y3, label='2012')

y4 = data[data['year'] == 2010]['Clean_Lyrics'].str.len()

sns.distplot(y4, label='2010')

y5 = data[data['year'] == 2008]['Clean_Lyrics'].str.len()

sns.distplot(y5, label='2008')

y6 = data[data['year'] == 2006]['Clean_Lyrics'].str.len()

sns.distplot(y6, label='2006')

plt.title('Year Wise - Lyrics Lenght Distribution (After Preprocessing)')

plt.legend();

data['year'].value_counts()
def ngram_extractor(text, n_gram):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



# Function to generate a dataframe with n_gram and top max_row frequencies

def generate_ngrams(df, col, n_gram, max_row):

    temp_dict = defaultdict(int)

    for question in df[col]:

        for word in ngram_extractor(question, n_gram):

            temp_dict[word] += 1

    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)

    temp_df.columns = ["word", "wordcount"]

    return temp_df



def comparison_plot(df_1,df_2,col_1,col_2, space):

    fig, ax = plt.subplots(1, 2, figsize=(20,10))

    

    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="skyblue")

    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="skyblue")



    ax[0].set_xlabel('Word count', size=14, color="green")

    ax[0].set_ylabel('Words', size=18, color="green")

    ax[0].set_title('Top words in 2017 Lyrics', size=18, color="green")



    ax[1].set_xlabel('Word count', size=14, color="green")

    ax[1].set_ylabel('Words', size=18, color="green")

    ax[1].set_title('Top words in 2008 Lyrics', size=18, color="green")



    fig.subplots_adjust(wspace=space)

    

    plt.show()
Lyrics_2017 = generate_ngrams(train[train["year"]==2017], 'Clean_Lyrics', 1, 10)

Lyrics_2008 = generate_ngrams(data[data["year"]==2008], 'Clean_Lyrics', 1, 10)

comparison_plot(Lyrics_2017,Lyrics_2008,'word','wordcount', 0.25)
Lyrics_2017 = generate_ngrams(train[train["year"]==2017], 'Clean_Lyrics', 2, 10)

Lyrics_2008 = generate_ngrams(data[data["year"]==2008], 'Clean_Lyrics', 2, 10)

comparison_plot(Lyrics_2017,Lyrics_2008,'word','wordcount', 0.25)
Lyrics_2017 = generate_ngrams(train[train["year"]==2017], 'Clean_Lyrics', 3, 10)

Lyrics_2008 = generate_ngrams(data[data["year"]==2008], 'Clean_Lyrics', 3, 10)

comparison_plot(Lyrics_2017,Lyrics_2008,'word','wordcount', 0.25)
import scattertext as st

nlp = spacy.load('en',disable_pipes=["tagger","ner"])

data['parsed'] = data.Clean_Lyrics.apply(nlp)

corpus = st.CorpusFromParsedDocuments(data,

                             category_col='album',

                             parsed_col='parsed').build()

html = st.produce_scattertext_explorer(corpus,

          category='reputation',

          category_name='reputation',

          not_category_name='1989',

          width_in_pixels=600,

          minimum_term_frequency=5,

          term_significance = st.LogOddsRatioUninformativeDirichletPrior(),

          )
filename = "reputation-vs-1989.html"

open(filename, 'wb').write(html.encode('utf-8'))

IFrame(src=filename, width = 800, height=700)
