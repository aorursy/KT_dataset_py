import numpy as np

import pandas as pd

import lightgbm as lgbm

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from skopt import gp_minimize

from skopt.space import Real, Integer

from skopt.plots import plot_convergence

import json

import string



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes,linear_model

from sklearn.ensemble import RandomForestClassifier



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



rand_seed = 13579

np.random.seed(rand_seed)



sns.set(style="darkgrid", context="notebook")



import os

print(os.listdir("../input"))

print(os.listdir("../input/train"))

print(os.listdir("../input/test"))
train_df = pd.read_csv("../input/train/train.csv")

test_df = pd.read_csv("../input/test/test.csv")
train_df.head()
test_df.head()
## Taken from SRKs Kernel

cnt_srs = train_df['AdoptionSpeed'].value_counts()

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

    

plot_wordcloud(train_df["Description"], title="Word Cloud of Train")
## Taken from SRKs Kernel

from wordcloud import WordCloud, STOPWORDS





def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    



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

    

plot_wordcloud(test_df["Description"], title="Word Cloud of Test")
train_df["Description"].fillna("#",inplace=True)
## Taken from SRKs Kernel

from collections import defaultdict

train1_df = train_df[train_df["AdoptionSpeed"]==0]

train2_df = train_df[train_df["AdoptionSpeed"]==1]

train3_df = train_df[train_df["AdoptionSpeed"]==2]

train4_df = train_df[train_df["AdoptionSpeed"]==3]

train5_df = train_df[train_df["AdoptionSpeed"]==4]





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

for sent in train1_df["Description"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')





freq_dict = defaultdict(int)

for sent in train2_df["Description"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train3_df["Description"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



freq_dict = defaultdict(int)

for sent in train4_df["Description"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(50), 'blue')





freq_dict = defaultdict(int)

for sent in train5_df["Description"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace4 = horizontal_bar_chart(fd_sorted.head(50), 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of Adoption Speed==0", 

                                          "Frequent words of Adoption Speed==1",

                                         "Frequent words of Adoption Speed==2",

                                         "Frequent words of Adoption Speed==3","Frequent words of Adoption Speed==4"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)





fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
## Taken from SRKs Kernel

freq_dict = defaultdict(int)

for sent in train1_df["Description"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in train2_df["Description"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in train3_df["Description"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in train4_df["Description"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



freq_dict = defaultdict(int)

for sent in train5_df["Description"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace4 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=3, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent bigrams of Adoption Speed==0", 

                                          "Frequent bigrams of Adoption Speed==1",

                                         "Frequent bigrams of Adoption Speed==2",

                                         "Frequent bigrams of Adoption Speed==3","Frequent bigrams of Adoption Speed==4"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 2, 2)

fig.append_trace(trace4, 3, 1)





fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
## Taken from SRKs Kernel

## Number of words in the text ##

train_df["num_words"] = train_df["Description"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["Description"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train_df["num_unique_words"] = train_df["Description"].apply(lambda x: len(set(str(x).split())))

test_df["num_unique_words"] = test_df["Description"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train_df["num_chars"] = train_df["Description"].apply(lambda x: len(str(x)))

test_df["num_chars"] = test_df["Description"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train_df["num_stopwords"] = train_df["Description"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

test_df["num_stopwords"] = test_df["Description"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



## Number of punctuations in the text ##

train_df["num_punctuations"] =train_df['Description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test_df["num_punctuations"] =test_df['Description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train_df["num_words_upper"] = train_df["Description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test_df["num_words_upper"] = test_df["Description"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train_df["num_words_title"] = train_df["Description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test_df["num_words_title"] = test_df["Description"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train_df["mean_word_len"] = train_df["Description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["Description"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
f, axes = plt.subplots(3, 1, figsize=(10,20))

sns.boxplot(x='AdoptionSpeed', y='num_words', data=train_df, ax=axes[0])

axes[0].set_xlabel('AdoptionSpeed', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='AdoptionSpeed', y='num_chars', data=train_df, ax=axes[1])

axes[1].set_xlabel('AdoptionSpeed', fontsize=12)

axes[1].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='AdoptionSpeed', y='num_punctuations', data=train_df, ax=axes[2])

axes[2].set_xlabel('AdoptionSpeed', fontsize=12)

axes[2].set_title("Number of punctuations in each class", fontsize=15)

plt.show()
# Get the tfidf vectors #

tfidf_vec = TfidfVectorizer(stop_words='english',ngram_range=(1,3),encoding='utf-8')

tfidf_vec.fit_transform(train_df['Description'].values.astype("U").tolist() + test_df['Description'].values.astype("U").tolist())

train_tfidf = tfidf_vec.transform(train_df['Description'].astype("U").values.tolist())

test_tfidf = tfidf_vec.transform(test_df['Description'].astype("U").values.tolist())
train_y = train_df["AdoptionSpeed"].values

model = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=2019)

model.fit(train_tfidf, train_y)
import eli5

eli5.show_weights(model, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')