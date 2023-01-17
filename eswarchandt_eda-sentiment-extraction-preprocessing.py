!pip install chart_studio

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

import re

import string



import matplotlib.pyplot as plt

from plotly import tools

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sub = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train['text'] = train['text'].str.replace('[{}]'.format(string.punctuation), '')

test['text'] = test['text'].str.replace('[{}]'.format(string.punctuation), '')
train.head(3)
test.head(3)
print(train.size)

print(train.shape)

print(test.shape)

print(test.size)
train.describe()
sns.countplot(train['sentiment'])
# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(10.0,10.0), color = 'white',

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=600, 

                    height=300,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(train.loc[train['sentiment'] == 'neutral', 'text'].append(test.loc[test['sentiment'] == 'neutral', 'text']), title="Word Cloud of Neutral tweets",color = 'white')
plot_wordcloud(train.loc[train['sentiment'] == 'positive', 'text'].append(test.loc[test['sentiment'] == 'positive', 'text']), title="Word Cloud of Positive tweets",color = 'green')
plot_wordcloud(train.loc[train['sentiment'] == 'negative', 'text'].append(test.loc[test['sentiment'] == 'negative', 'text']), title="Word Cloud of negative tweets",color = 'red')
from collections import defaultdict

train0_df = train[train["sentiment"]=='positive'].dropna().append(test[test["sentiment"]=='positive'].dropna())

train1_df = train[train["sentiment"]=='neutral'].dropna().append(test[test["sentiment"]=='neutral'].dropna())

train2_df = train[train["sentiment"]=='negative'].dropna().append(test[test["sentiment"]=='neutral'].dropna())



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

for sent in train0_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'red')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'green')



## Get the bar chart from sincere questions ##

freq_dict = defaultdict(int)

for sent in train2_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of positive tweets", "Frequent words of neutral tweets",

                                          "Frequent words of negative tweets"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'gray')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'orange')



freq_dict = defaultdict(int)

for sent in train2_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'brown')







# Creating two subplots

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,horizontal_spacing=0.25,

                          subplot_titles=["Bigram plots of Positive tweets", 

                                          "Bigram plots of Neutral tweets",

                                          "Bigram plots of Negative tweets"

                                          ])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)





fig['layout'].update(height=1000, width=800, paper_bgcolor='rgb(233,233,233)', title="Bigram Plots")

iplot(fig, filename='word-plots')
for sent in train0_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(25), 'blue')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(25), 'green')



freq_dict = defaultdict(int)

for sent in train2_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(25), 'violet')









# Creating two subplots

fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04, horizontal_spacing=0.05,

                          subplot_titles=["Tri-gram plots of Positive tweets", 

                                          "Tri-gram plots of Neutral tweets",

                                          "Tri-gram plots of Negative tweets"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

iplot(fig, filename='word-plots')
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
fig = go.Figure()

fig.add_trace(go.Histogram(x=train['num_words'],name = 'Number of words in text of train data'))

fig.add_trace(go.Histogram(x=test['num_words'],name = 'Number of words in text of test data'))

fig.add_trace(go.Histogram(x=train['select_num_words'],name = 'Number of words in selected text'))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig_ = go.Figure()

fig_.add_trace(go.Histogram(x=train['num_chars'],name = 'Number of characters in text of train data',marker = dict(color = 'rgba(222, 111, 33, 0.8)')))

fig_.add_trace(go.Histogram(x=test['num_chars'],name = 'Number of characters in text of test data',marker = dict(color = 'rgba(33, 1, 222, 0.8)')))

fig_.add_trace(go.Histogram(x=train['select_num_chars'],name = 'Number of characters in selected text',marker = dict(color = 'rgba(108, 25, 7, 0.8)')))



# Overlay both histograms

fig_.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig_.update_traces(opacity=0.75)

fig_.show()
fig_ = go.Figure()

fig_.add_trace(go.Histogram(x=train['num_unique_words'],name = 'Number of unique words in text of train data',marker = dict(color = 'rgba(222, 1, 3, 0.8)')))

fig_.add_trace(go.Histogram(x=test['num_unique_words'],name = 'Number of unique words in text of test data',marker = dict(color = 'rgba(3, 221, 2, 0.8)')))

fig_.add_trace(go.Histogram(x=train['select_num_unique_words'],name = 'Number of unique words in selected text',marker = dict(color = 'rgba(1, 2, 237, 0.8)')))



# Overlay both histograms

fig_.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig_.update_traces(opacity=0.75)

fig_.show()