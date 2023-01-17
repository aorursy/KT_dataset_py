# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





#train_char = pd.read_csv('../input/train.csv')

#test_char = pd.read_csv('../input/test.csv')
train.head()
from collections import defaultdict

from nltk.corpus import stopwords

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



stop_words = set(stopwords.words('english')) 

insinc_df = train[train.target==1]

sinc_df = train[train.target==0]



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

        trace = horizontal_bar_chart(fd_sorted.head(10), bar_color)

        return trace    



    trace0 = get_bar(sinc_df, 'blue')

    trace1 = get_bar(insinc_df, 'blue')



    # Creating two subplots

    if n_grams == 1:

        wrd = "words"

    elif n_grams == 2:

        wrd = "bigrams"

    elif n_grams == 3:

        wrd = "trigrams"

    

    fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                              subplot_titles=["Frequent " + wrd + " of sincere questions", 

                                              "Frequent " + wrd + " of insincere questions"])

    fig.append_trace(trace0, 1, 1)

    fig.append_trace(trace1, 1, 2)

    fig['layout'].update(height=500, width=1150, paper_bgcolor='rgb(233,233,233)', title=wrd + " Count Plots")

    py.iplot(fig, filename='word-plots')
plot_ngrams(1)
plot_ngrams(2)
plot_ngrams(3)
#######################

#EDA for training Data#

#######################



t1 = train[["target"]]

def score_to_numeric(x):

    if x==0:

        return "Sincere"

    if x==1:

        return "Insincere"

t1['target_class'] = t1['target'].apply(score_to_numeric)

t1_1 = t1.groupby(['target_class']).count()

t1_1
########################

#number of group target#

########################



import plotly.graph_objects as go

fig = go.Figure(go.Bar(

            x=[80810, 1225312],

            y=['Insincere', 'Sincere'],

            orientation='h',

    marker=dict(

        color='rgba(51, 204, 204, 0.6)',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=3))))



fig.show()
import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer

####################################

#number of character in each target#

####################################

t2 = train[["question_text", "target"]]

def score_to_numeric(x):

    if x==0:

        return "Sincere"

    if x==1:

        return "Insincere"

t2['target_class'] = t2['target'].apply(score_to_numeric)

t2_1 = t2[['question_text', 'target_class']]

t2_1.head()
def tokenize_character(text):

    text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower()   

    text = re.sub(r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*', ' ', text) # remove hyperlink,subs charact in the brackets

    text = re.sub("[\r\n]", ' ', text) # remove new line characters

    tokens = word_tokenize(text)

    tokens = [token for token in tokens if re.match(r'.*[a-z]{2,}.*', token)]

    return tokens



#t2_1["tokens"] = t2_1['question_text'].map(lambda x: tokenize_character(x))

t2_1["tokens"] = t2_1['question_text'].apply(tokenize_character) 

t2_1["word_count"] = t2_1.tokens.apply(lambda x: len(x))

t2_1.tail()
t2_2 = t2_1[["target_class", "word_count"]]

t2_2.head()
# library & dataset

import seaborn as sns

#df = sns.load_dataset('iris')

sns.boxplot(x=t2_2["target_class"], y=t2_2["word_count"])

#sns.plt.show()