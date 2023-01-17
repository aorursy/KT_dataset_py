import os

import json

import string

import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb



import plotly.offline

#import cufflinks as cf

#cf.go_offline()

#cf.set_config_file(offline=False, world_readable=True)





pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
!ls ../input/
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)
train_df.head()
train_df["target2"]=(train_df["target"]>0.5)
target1=[]

for i,row in train_df.iterrows():

    if row["target2"]==False:

        target1.append(0)

    else:

        target1.append(1)  
train_df["final_target"]=target1
import re

corpus = []

for i in range(0, train_df.shape[0]):

    review = re.sub('[.]', '', train_df['comment_text'][i])

    corpus.append(review)
train_df['comment_text']=corpus
## Taken from SRKs Kernel

## target count ##

cnt_srs = train_df['final_target'].value_counts()

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



## target distribution ##

labels = (np.array(cnt_srs.index))

sizes = (np.array((cnt_srs / cnt_srs.sum())*100))



trace = go.Pie(labels=labels, values=sizes)

layout = go.Layout(

    title='Target distribution',

    font=dict(size=18),

    width=600,

    height=600,

)

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="usertype")
## Taken from SRKs Kernel

from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

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

    

plot_wordcloud(train_df["comment_text"], title="Word Cloud of Comments")
## Taken from SRKs Kernel

from collections import defaultdict

train1_df = train_df[train_df["final_target"]==1]

train0_df = train_df[train_df["final_target"]==0]



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



## Get the bar chart from non-toxic comments ##

freq_dict = defaultdict(int)

for sent in train0_df["comment_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



## Get the bar chart from toxic comments ##

freq_dict = defaultdict(int)

for sent in train1_df["comment_text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of non-toxic comments", 

                                          "Frequent words of toxic comments"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



#plt.figure(figsize=(10,16))

#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")

#plt.title("Frequent words for Insincere Questions", fontsize=16)

#plt.show()
## Taken from SRKs Kernel

freq_dict = defaultdict(int)

for sent in train0_df["comment_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in train1_df["comment_text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent bigrams of non-toxic comments", 

                                          "Frequent bigrams of toxic comments"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
## Taken from SRKs Kernel

freq_dict = defaultdict(int)

for sent in train0_df["comment_text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'green')





freq_dict = defaultdict(int)

for sent in train1_df["comment_text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,

                          subplot_titles=["Frequent trigrams of non-toxic comments", 

                                          "Frequent trigrams of toxic comments"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

py.iplot(fig, filename='word-plots')
## Taken from SRKs Kernel

## Number of words in the text ##

train_df["num_words"] = train_df["comment_text"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["comment_text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train_df["num_unique_words"] = train_df["comment_text"].apply(lambda x: len(set(str(x).split())))

test_df["num_unique_words"] = test_df["comment_text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train_df["num_chars"] = train_df["comment_text"].apply(lambda x: len(str(x)))

test_df["num_chars"] = test_df["comment_text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train_df["num_stopwords"] = train_df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

test_df["num_stopwords"] = test_df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



## Number of punctuations in the text ##

train_df["num_punctuations"] =train_df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test_df["num_punctuations"] =test_df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train_df["num_words_upper"] = train_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test_df["num_words_upper"] = test_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train_df["num_words_title"] = train_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test_df["num_words_title"] = test_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train_df["mean_word_len"] = train_df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
## Taken from SRKs Kernel

## Truncate some extreme values for better visuals ##

train_df['num_words'].loc[train_df['num_words']>60] = 60 #truncation for better visuals

train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 #truncation for better visuals

train_df['num_chars'].loc[train_df['num_chars']>350] = 350 #truncation for better visuals



f, axes = plt.subplots(3, 1, figsize=(10,20))

sns.boxplot(x='final_target', y='num_words', data=train_df, ax=axes[0])

axes[0].set_xlabel('Target', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='final_target', y='num_chars', data=train_df, ax=axes[1])

axes[1].set_xlabel('Target', fontsize=12)

axes[1].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='final_target', y='num_punctuations', data=train_df, ax=axes[2])

axes[2].set_xlabel('Target', fontsize=12)

#plt.ylabel('Number of punctuations in text', fontsize=12)

axes[2].set_title("Number of punctuations in each class", fontsize=15)

plt.show()
plt.figure(figsize=(8, 6))

sns.heatmap(train_df.select_dtypes(include=['float64']).corr(),cmap="YlGnBu")