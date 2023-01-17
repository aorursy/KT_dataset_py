# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

import os

import re

import csv

import string

import gc

from tqdm import tqdm



import numpy as np

import pandas as pd

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from scipy.sparse import hstack

from IPython.display import Image

from prettytable import PrettyTable



from tqdm import tqdm_notebook

tqdm_notebook().pandas()



from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from nltk.stem.lancaster import LancasterStemmer

from nltk.util import ngrams
values = [df_train[df_train['target']==0].shape[0], df_train[df_train['target']==1].shape[0]]

labels = ['Non Toxic questions', 'Toxic questions']



plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=True)

plt.title('Target Distribution')

plt.tight_layout()

plt.subplots_adjust(right=1.9)

plt.show()
print(df_train.target.value_counts())
cnt_srs = df_train['target'].value_counts()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color=cnt_srs.values,

        colorscale = 'Picnic',

        reversescale = True

    ),

)



layout = go.Layout(

    title='Target Count',

    font=dict(size=18)

)



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="TargetCount")


# Number of words

df_train['num_words'] = df_train['question_text'].apply(lambda x: len(str(x).split()))

#df_test['num_words'] = df_test['question_text'].apply(lambda x: len(str(x).split()))



# Number of capital_letters

df_train['num_capital_let'] = df_train['question_text'].apply(lambda x: len([c for c in str(x) if c.isupper()]))

#df_test['num_capital_let'] = df_test['question_text'].apply(lambda x: len([c for c in str(x) if c.isupper()]))



# Number of special characters

df_train['num_special_char'] = df_train['question_text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()

#df_test['num_special_char'] = df_test['question_text'].str.findall(r'[^a-zA-Z0-9 ]').str.len()



# Number of unique words

df_train['num_unique_words'] = df_train['question_text'].apply(lambda x: len(set(str(x).split())))

#df_test['num_unique_words'] = df_test['question_text'].apply(lambda x: len(set(str(x).split())))



# Number of numerics

df_train['num_numerics'] = df_train['question_text'].apply(lambda x: sum(c.isdigit() for c in x))

#df_test['num_numerics'] = df_test['question_text'].apply(lambda x: sum(c.isdigit() for c in x))



# Number of characters

df_train['num_char'] = df_train['question_text'].apply(lambda x: len(str(x)))

#df_test['num_char'] = df_test['question_text'].apply(lambda x: len(str(x)))



# Number of stopwords

df_train['num_stopwords'] = df_train['question_text'].apply(lambda x: len([c for c in str(x).lower().split() if c in STOPWORDS]))

#df_test['num_stopwords'] = df_test['question_text'].apply(lambda x: len([c for c in str(x).lower().split() if c in STOPWORDS]))

def display_boxplot(_x, _y, _data, _title):

    sns.boxplot(x=_x, y=_y, data=_data)

    plt.grid(True)

    #plt.tick_params(axis='x', which='major', labelsize=15)

    plt.title(_title,fontsize=17)

    plt.xlabel(_x, fontsize=10)



# Boxplot: Number of words

plt.subplot(2, 3, 1)

display_boxplot('target', 'num_words', df_train, 'No. of words in each class')



# Boxplot: Number of chars

plt.subplot(2, 3, 2)

display_boxplot('target', 'num_char', df_train, 'Number of characters in each class')



# Boxplot: Number of unique words

plt.subplot(2, 3, 3)

display_boxplot('target', 'num_unique_words', df_train, 'Number of unique words in each class')



# Boxplot: Number of special characters

plt.subplot(2, 3, 4)

display_boxplot('target', 'num_special_char', df_train, 'No. of special characters in each class')



# Boxplot: Number of stopwords

plt.subplot(2, 3, 5)

display_boxplot('target', 'num_stopwords', df_train, 'Number of stopwords in each class')



# Boxplot: Number of capital letters

plt.subplot(2, 3, 6)

display_boxplot('target', 'num_capital_let', df_train, 'No. of capital letters in each class')





plt.subplots_adjust(right=3.0)

plt.subplots_adjust(top=2.0)

plt.show()
# Correlation matrix

f, ax = plt.subplots(figsize=(10, 8))

corr = df_train.corr()

sns.heatmap(corr, ax=ax,annot=True)

plt.title("Correlation matrix")

plt.show()
from collections import defaultdict

train1_df = df_train[df_train["target"]==1]

train0_df = df_train[df_train["target"]==0]



## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(10), 'orange')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(10), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of Non-toxic Questions", 

                                          "Frequent words of  Toxic questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=500, width=750, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='word-plots')
freq_dict = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(10), 'orange')





freq_dict = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(10), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams of Non-Toxic Questions", 

                                          "Frequent bigrams of Toxic Questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=500, width=750, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='word-plots')
freq_dict = defaultdict(int)

for sent in train0_df["question_text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(10), 'orange')





freq_dict = defaultdict(int)

for sent in train1_df["question_text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(10), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,

                          subplot_titles=["Frequent trigrams of Non-Toxic ", 

                                          "Frequent trigrams of Toxic Questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=500, width=800, paper_bgcolor='rgb(233,233,233)')

py.iplot(fig, filename='word-plots')
from collections import defaultdict

from nltk.corpus import stopwords

from nltk import WordNetLemmatizer

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

stop_words = set(stopwords.words('english')) 

insinc_df = df_train[df_train.target==1]

sinc_df = df_train[df_train.target==0]

def plot_ngrams(n_grams):

    ## custom function for ngram generation ##

    def generate_ngrams(text, n_gram=1):

        token = [token for token in text.lower().split(" ") if token != "" if token not in stop_words]

        ngrams = zip(*[token[i:] for i in range(n_gram)])

        return [" ".join(ngram) for ngram in ngrams]

    ## custom function for horizontal bar chart ##

    def horizontal_bar_chart(df, color):

        trace = go.Bar(

            y=df["word"].values[::-1],

            x=df["wordcount"].values[::-1],

            showlegend=False,

            orientation = 'h',

            marker=dict(

                color=color,

            ),

        )

        return trace

    def get_bar(df, bar_color):

        freq_dict = defaultdict(int)

        for sent in df["question_text"]:

            for word in generate_ngrams(sent, n_grams):

                freq_dict[word] += 1

        fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

        fd_sorted.columns = ["word", "wordcount"]

        trace = horizontal_bar_chart(fd_sorted.head(13), bar_color)

        return trace    

    trace0 = get_bar(sinc_df, 'green')

    trace1 = get_bar(insinc_df, 'red')

    # Creating two subplots

    if n_grams == 1:

        wrd = "words"

    elif n_grams == 2:

        wrd = "Bigrams"

    elif n_grams == 3:

        wrd = "Trigrams"    

    fig = toolsmake_subplots(rows=1, cols=2, vertical_spacing=0.03,subplot_titles=["Frequent " + wrd + " of Toxic", 

            "Frequent " + wrd + " of Non-toxic "])

    fig.append_trace(trace0, 1, 1)

    fig.append_trace(trace1, 1, 2)

    fig['layout'].update(height=500, width=750, paper_bgcolor='rgb(233,233,233)', title=wrd + " Count Plots")

    py.iplot(fig, filename='word-plots')
#Unigram

plot_ngrams(1)

#Bigram

plot_ngrams(2)

#Trigram

plot_ngrams(3)