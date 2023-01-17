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
!pip install textstat
import string

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from statistics import *

import concurrent.futures

import time

import pyLDAvis.sklearn

from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

import textstat



import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter

import folium 

from folium import plugins 





#utility functions:

def plot_readability(a,b,title,bins=0.1,colors=['#3A4750', '#F64E8B']):

    trace1 = ff.create_distplot([a,b], [" Real disaster tweets","Not real disaster tweets"], bin_size=bins, colors=colors, show_rug=False)

    trace1['layout'].update(title=title)

    py.iplot(trace1, filename='Distplot')

    table_data= [["Statistical Measures"," Not real disaster tweets","real disaster tweets"],

                ["Mean",mean(a),mean(b)],

                ["Standard Deviation",pstdev(a),pstdev(b)],

                ["Variance",pvariance(a),pvariance(b)],

                ["Median",median(a),median(b)],

                ["Maximum value",max(a),max(b)],

                ["Minimum value",min(a),min(b)]]

    trace2 = ff.create_table(table_data)

    py.iplot(trace2, filename='Table')



punctuations = string.punctuation

stopwords = list(STOP_WORDS)



parser = English()

def spacy_tokenizer(sentence):

    mytokens = parser(sentence)

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]

    mytokens = " ".join([i for i in mytokens])

    return mytokens



import re

def cleanhtml(raw_html):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', raw_html)

    return cleantext



def removeurl(raw_text):

    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', raw_text, flags=re.MULTILINE)

    return clean_text
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
#glimpse at train dataset

train.head()
#glimpse at test dataset

test.head()
#some basic cleaning

train['text'] = train['text'].apply(lambda x:cleanhtml(x))

test['text'] = test['text'].apply(lambda x:cleanhtml(x))



#removing url tags

train['text'] = train['text'].apply(lambda x:removeurl(x))

test['text'] = test['text'].apply(lambda x:removeurl(x))

cnt_srs = train['target'].value_counts()

trace = go.Bar(

    x=cnt_srs.index,

    y=cnt_srs.values,

    marker=dict(

        color=cnt_srs.values,

        colorscale = 'Jet',

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
cnt_ = train['location'].value_counts()

cnt_.reset_index()

cnt_ = cnt_[:20,]

trace1 = go.Bar(

                x = cnt_.index,

                y = cnt_.values,

                name = "Number of tweets in dataset according to location",

                marker = dict(color = 'rgba(200, 74, 55, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                )



data = [trace1]

layout = go.Layout(barmode = "group",title = 'Number of tweets in dataset according to location')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
train1_df = train[train["target"]==1]

train0_df = train[train["target"]==0]

cnt_1 = train1_df['location'].value_counts()

cnt_1.reset_index()

cnt_1 = cnt_1[:20,]



cnt_0 = train0_df['location'].value_counts()

cnt_0.reset_index()

cnt_0 = cnt_0[:20,]



trace1 = go.Bar(

                x = cnt_1.index,

                y = cnt_1.values,

                name = "Number of tweets about real disaster location wise",

                marker = dict(color = 'rgba(255, 74, 55, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                )

trace0 = go.Bar(

                x = cnt_0.index,

                y = cnt_0.values,

                name = "Number of tweets other than real disaster location wise",

                marker = dict(color = 'rgba(79, 82, 97, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                )





data = [trace0,trace1]

layout = go.Layout(barmode = 'stack',title = 'Number of tweets in dataset according to location')

fig = go.Figure(data = data, layout = layout)

py.iplot(fig)
df = train['location'].value_counts()[:20,]

df = pd.DataFrame(df)

df = df.reset_index()

df.columns = ['location', 'counts'] 

geolocator = Nominatim(user_agent="specify_your_app_name_here")

geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

dictt_latitude = {}

dictt_longitude = {}

for i in df['location'].values:

    print(i)

    location = geocode(i)

    dictt_latitude[i] = location.latitude

    dictt_longitude[i] = location.longitude

df['latitude']= df['location'].map(dictt_latitude)

df['longitude'] = df['location'].map(dictt_longitude)
map1 = folium.Map(location=[10.0, 10.0], tiles='CartoDB dark_matter', zoom_start=2.3)

markers = []

for i, row in df.iterrows():

    loss = row['counts']

    if row['counts'] > 0:

        count = row['counts']*0.4

    folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=float(count), color='#ef4f61', fill=True).add_to(map1)

map1
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

    

plot_wordcloud(train[train["target"]==1], title="Word Cloud of tweets if real disaster")
plot_wordcloud(train[train["target"]==0], title="Word Cloud of tweets if not a real disaster")
from collections import defaultdict

train1_df = train[train["target"]==1]

train0_df = train[train["target"]==0]



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

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'red')



## Get the bar chart from insincere questions ##

freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'red')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words if tweet is not real disaster", 

                                          "Frequent words if tweet is real disaster"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'green')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,2):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,

                          subplot_titles=["Frequent words if tweet is not real disaster", 

                                          "Frequent words if tweet is real disaster"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
freq_dict = defaultdict(int)

for sent in train0_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')





freq_dict = defaultdict(int)

for sent in train1_df["text"]:

    for word in generate_ngrams(sent,3):

        freq_dict[word] += 1

fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,

                          subplot_titles=["Frequent words if tweet is not real disaster", 

                                          "Frequent words if tweet is real disaster"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

py.iplot(fig, filename='word-plots')
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train["num_unique_words"] = train["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train["num_chars"] = train["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



## Number of punctuations in the text ##

train["num_punctuations"] =train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train["num_words_upper"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train["num_words_title"] = train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train["mean_word_len"] = train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["mean_word_len"] = test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train['num_words'].loc[train['num_words']>60] = 100 #truncation for better visuals

train['num_punctuations'].loc[train['num_punctuations']>25] = 25 #truncation for better visuals

train['num_chars'].loc[train['num_chars']>350] = 350 #truncation for better visuals



f, axes = plt.subplots(3, 1, figsize=(10,20))

sns.boxplot(x='target', y='num_words', data=train, ax=axes[0])

axes[0].set_xlabel('Target', fontsize=12)

axes[0].set_title("Number of words in each class", fontsize=15)



sns.boxplot(x='target', y='num_chars', data=train, ax=axes[1])

axes[1].set_xlabel('Target', fontsize=12)

axes[1].set_title("Number of characters in each class", fontsize=15)



sns.boxplot(x='target', y='num_punctuations', data=train, ax=axes[2])

axes[2].set_xlabel('Target', fontsize=12)

axes[2].set_title("Number of punctuations in each class", fontsize=15)

plt.show()
train1_df = train[train["target"]==1]

train0_df = train[train["target"]==0]



fig = go.Figure()

fig.add_trace(go.Histogram(x=train1_df['num_words'],name = 'Number of words in tweets about real disaster'))

fig.add_trace(go.Histogram(x=train0_df['num_words'],name = 'Number of words in tweets other than real disaster'))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=train1_df['num_chars'],name = 'Number of chars in tweets about real disaster',marker = dict(color = 'rgba(200, 100, 0, 0.8)')))

fig.add_trace(go.Histogram(x=train0_df['num_chars'],name = 'Number of chars in tweets about real disaster',marker = dict(color = 'rgba(25, 133, 120, 0.8)')))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=train1_df['num_punctuations'],name = 'Number of punctuations in tweets about real disaster',marker = dict(color = 'rgba(97, 175, 222, 0.8)')))

fig.add_trace(go.Histogram(x=train0_df['num_punctuations'],name = 'Number of punctuations in tweets other than real disaster',marker = dict(color = 'rgba(200, 10, 150, 0.8)')))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=1)

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=train['num_words'],name = 'Number of words in training tweets',marker = dict(color = 'rgba(255, 0, 0, 0.8)')))

fig.add_trace(go.Histogram(x=test['num_words'],name = 'Number of words in testing tweets ',marker = dict(color = 'rgba(0, 187, 187, 0.8)')))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=train['num_chars'],name = 'Number of chars in training tweets',marker = dict(color = 'rgba(25, 13, 8, 0.8)')))

fig.add_trace(go.Histogram(x=test['num_chars'],name = 'Number of chars in testing tweets ',marker = dict(color = 'rgba(8, 25, 187, 0.8)')))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=train['num_punctuations'],name = 'Number of punctuations in training tweets',marker = dict(color = 'rgba(222, 111, 33, 0.8)')))

fig.add_trace(go.Histogram(x=test['num_punctuations'],name = 'Number of punctuations in testing tweets ',marker = dict(color = 'rgba(33, 111, 222, 0.8)')))



# Overlay both histograms

fig.update_layout(barmode='stack')

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
tqdm.pandas()

fre_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.flesch_reading_ease))

fre_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.flesch_reading_ease))

plot_readability(fre_notreal,fre_real,"Flesch Reading Ease",20)
fkg_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.flesch_kincaid_grade))

fkg_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.flesch_kincaid_grade))

plot_readability(fkg_notreal,fkg_real,"Flesch Kincaid Grade",4,['#C1D37F','#491F21'])
fog_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.gunning_fog))

fog_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.gunning_fog))

plot_readability(fog_notreal,fog_real,"The Fog Scale (Gunning FOG Formula)",4,['#E2D58B','#CDE77F'])
ari_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.automated_readability_index))

ari_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.automated_readability_index))

plot_readability(ari_notreal,ari_real,"Automated Readability Index",10,['#488286','#FF934F'])
cli_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.coleman_liau_index))

cli_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.coleman_liau_index))

plot_readability(cli_notreal,cli_real,"The Coleman-Liau Index",10,['#8491A3','#2B2D42'])
lwf_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.linsear_write_formula))

lwf_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.linsear_write_formula))

plot_readability(lwf_notreal,lwf_real,"Linsear Write Formula",2,['#8D99AE','#EF233C'])
dcr_notreal = np.array(train["text"][train["target"] == 0].progress_apply(textstat.dale_chall_readability_score))

dcr_real = np.array(train["text"][train["target"] == 1].progress_apply(textstat.dale_chall_readability_score))

plot_readability(dcr_notreal,dcr_real,"Dale-Chall Readability Score",1,['#C65D17','#DDB967'])
def consensus_all(text):

    return textstat.text_standard(text,float_output=True)



con_notreal = np.array(train["text"][train["target"] == 0].progress_apply(consensus_all))

con_real = np.array(train["text"][train["target"] == 1].progress_apply(consensus_all))

plot_readability(con_notreal,con_real,"Readability Consensus based upon all the above tests",2)
notreal_text = train["text"][train["target"] == 0].progress_apply(spacy_tokenizer)

real_text = train["text"][train["target"] == 1].progress_apply(spacy_tokenizer)

#count vectorization

vectorizer_notreal = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

notreal_vectorized = vectorizer_notreal.fit_transform(notreal_text)

vectorizer_real = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

real_vectorized = vectorizer_real.fit_transform(real_text)
# Latent Dirichlet Allocation Model

lda_notreal = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)

notreal_lda = lda_notreal.fit_transform(notreal_vectorized)

lda_real = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)

real_lda = lda_real.fit_transform(real_vectorized)
def selected_topics(model, vectorizer, top_n=10):

    for idx, topic in enumerate(model.components_):

        print("Topic %d:" % (idx))

        print([(vectorizer.get_feature_names()[i], topic[i])

                        for i in topic.argsort()[:-top_n - 1:-1]]) 
print("Not real disaster tweets LDA Model:")

selected_topics(lda_notreal, vectorizer_notreal)
print("Real disaster tweets LDA Model:")

selected_topics(lda_real, vectorizer_real)
pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_notreal, notreal_vectorized, vectorizer_notreal, mds='tsne')

dash
pyLDAvis.enable_notebook()

dash = pyLDAvis.sklearn.prepare(lda_real, real_vectorized, vectorizer_real, mds='tsne')

dash
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

tfidf_vec.fit_transform(train['text'].values.tolist() + test['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test['text'].values.tolist())
train_y = train["target"].values



def runModel(train_X, train_y, test_X, test_y, test_X2):

    model = linear_model.LogisticRegression(C=5., solver='sag')

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)[:,1]

    pred_test_y2 = model.predict_proba(test_X2)[:,1]

    return pred_test_y, pred_test_y2, model



print("Building model.")

cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train.shape[0]])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    break
from tqdm import tqdm

def threshold_search(y_true, y_proba):

#reference: https://www.kaggle.com/hung96ad/pytorch-starter

    best_threshold = 0

    best_score = 0

    for threshold in tqdm([i * 0.001 for i in range(1000)]):

        score = metrics.f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result

search_result = threshold_search(val_y, pred_val_y)

search_result
print("F1 score at threshold {0} is {1}".format(0.381, metrics.f1_score(val_y, (pred_val_y>0.381).astype(int))))

print("Precision at threshold {0} is {1}".format(0.381, metrics.precision_score(val_y, (pred_val_y>0.381).astype(int))))

print("recall score at threshold {0} is {1}".format(0.381, metrics.recall_score(val_y, (pred_val_y>0.381).astype(int))))
import eli5

eli5.show_weights(model, vec=tfidf_vec, top=100, feature_filter=lambda x: x != '<BIAS>')
#loading markovify module

import markovify
#preparing dataset

data_notreal = train["text"][train["target"] == 0]

data_real = train["text"][train["target"] == 1]
text_model_notreal = markovify.NewlineText(data_notreal, state_size = 2)

text_model_real = markovify.NewlineText(data_real, state_size = 2)
for i in range(10):

    print(text_model_notreal.make_sentence())
for i in range(10):

    print(text_model_real.make_sentence())