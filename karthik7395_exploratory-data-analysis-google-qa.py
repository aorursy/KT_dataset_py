# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

import string

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")

test_df = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")

print("Train shape : ", train_df.shape)

print("Test shape : ", test_df.shape)
train_df.head().T
test_df.head()
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

    

plot_wordcloud(train_df["question_title"], title="Word Cloud for question title")
from collections import defaultdict





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



freq_dict = defaultdict(int)

for sent in train_df["question_title"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace = horizontal_bar_chart(fd_sorted.head(50), 'blue')



fig = go.Figure(data=[trace])



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
plot_wordcloud(train_df["question_body"], title="Word Cloud for question body")
freq_dict = defaultdict(int)

for sent in train_df["question_body"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace = horizontal_bar_chart(fd_sorted.head(50), 'blue')



fig = go.Figure(data=[trace])



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
fig = go.Figure(data=[go.Histogram(x=train_df['question_user_name'].value_counts())])

fig.show()
fig = go.Figure(data=[go.Histogram(x=train_df['host'])])

fig.show()
fig = go.Figure(data=[go.Histogram(x=train_df['category'])])

fig.show()
train_df['answer_user_name'].value_counts()
fig = go.Figure(data=[go.Histogram(x=train_df['answer_user_name'].value_counts())])

fig.show()
plot_wordcloud(train_df["answer"], title="Word Cloud for answer")
freq_dict = defaultdict(int)

for sent in train_df["answer"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace = horizontal_bar_chart(fd_sorted.head(50), 'blue')



fig = go.Figure(data=[trace])



fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
target_cols=['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
fig = subplots.make_subplots(rows=6, cols=5, vertical_spacing=0.06,

                          subplot_titles=target_cols)

for ind,col  in enumerate(target_cols):

    dist_dict = pd.DataFrame(sorted(train_df[col].value_counts().to_dict().items(), key=lambda x: x[1])[::-1])

    trace = go.Bar(

        y=dist_dict[0],

        x=dist_dict[1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color='blue',

        )

    )  

    fig.append_trace(trace, ind//5 + 1 , ind%5 + 1) 

fig['layout'].update(height=1200, width=1700, paper_bgcolor='rgb(233,233,233)', title="target dist plot")

py.iplot(fig, filename='target-plots')