# Import Libraries

import glob
import json
import datetime as dt
import os
import string
import re    #for regex

import random

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# plotting
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import cm
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py

# natural language processing
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

import nltk
from nltk.util import bigrams
from nltk.util import trigrams
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer, TweetTokenizer

from stop_words import get_stop_words
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import operator
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from statistics import *
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig

# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 

# set seeds for reproducability
import tensorflow as tf
from numpy.random import seed
tf.random.set_seed(2)
seed(1)

import warnings
warnings.filterwarnings("ignore")
#Setting the stopwords to later remove all the stopwords from the text
spacey_words = list(STOP_WORDS)       
nltk_words = list(stopwords.words('english'))   
stop_words = list(get_stop_words('en'))  
stop_words.extend(nltk_words)
stop_words.extend(spacey_words)
eng_stop_words = set(stopwords.words("english"))
# Remove certain popular junk terms for clearer data
stop_words.extend(['com', 'video', 'Video', 'youtube', 'http', 'www', 'https', 'nhttp', 'nhttps', 'ntwitter', 'nfacebook', 'ninstagram', 'the'])
eng_stopwords = eng_stop_words.union(set(stop_words))
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
csv_files = [i for i in glob.glob('/kaggle/input/youtube-new/*.{}'.format('csv'))]
sorted(csv_files)
json_files = [i for i in glob.glob('/kaggle/input/youtube-new/*.{}'.format('json'))]
sorted(json_files)
countries = []
for csv in csv_files:
    country_df = pd.read_csv(csv, index_col='video_id', encoding='ISO-8859-1')
    country_df['country'] = csv[26:28]
    countries.append(country_df)

yt_df = pd.concat(countries)
# yt_df.head(5)
yt_df['category_id'] = yt_df['category_id'].astype(str)

category_id = {}

with open('/kaggle/input/youtube-new/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        category_id[category['id']] = category['snippet']['title']

yt_df.insert(4, 'category', yt_df['category_id'].map(category_id))

categories = yt_df[['category_id', 'category']].drop_duplicates().sort_values('category_id')

categories
print('Shape of dataframe:')
print(yt_df.shape)
print('\n\nUnique entries:\n')
print(yt_df.nunique())
print('\n\nDataframe info():\n') 
yt_df.info()
yt_df.head(5)
yt_df['trending_date'] = pd.to_datetime(yt_df['trending_date'], errors='coerce', format='%y.%d.%m').dt.date
yt_df['publish_time'] = pd.to_datetime(yt_df['publish_time'], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')

yt_df = yt_df[yt_df['trending_date'].notnull()]
yt_df = yt_df[yt_df['publish_time'].notnull()]

yt_df = yt_df.dropna(how='any', inplace=False, axis = 0)

yt_df.insert(4, 'publish_date', yt_df['publish_time'].dt.date)
yt_df['publish_hour'] = yt_df['publish_time'].dt.hour
yt_df['publish_time'] = yt_df['publish_time'].dt.time

yt_df_complete = yt_df.reset_index().sort_values('trending_date').set_index('video_id')
yt_df = yt_df.reset_index().sort_values('trending_date').drop_duplicates('video_id',keep='last').set_index('video_id')

yt_df['time_to_trend'] = (yt_df['trending_date'] - yt_df['publish_date']) / np.timedelta64(1, 'D')

yt_df[['trending_date', 'time_to_trend', 'publish_date', 'publish_time', 'publish_hour']].head()
# Participation percentages and ratios
yt_df['dislike_percentage'] = yt_df['dislikes'] / (yt_df['dislikes'] + yt_df['likes'])
yt_df['like_dislike_ratio'] = yt_df['likes'] / yt_df['dislikes']
yt_df['comment_percent'] = yt_df['comment_count'] / yt_df['views']
yt_df['reaction_percent'] = (yt_df['likes'] + yt_df['dislikes']) / yt_df['views']
# Participation logs
yt_df['likes_log'] = np.log(yt_df['likes'] + 1)
yt_df['views_log'] = np.log(yt_df['views'] + 1)
yt_df['dislikes_log'] = np.log(yt_df['dislikes'] + 1)
yt_df['comment_log'] = np.log(yt_df['comment_count'] + 1)
# Participation rate
yt_df['like_rate'] =  yt_df ['likes'] / yt_df['views'] * 100
yt_df['dislike_rate'] =  yt_df ['dislikes'] / yt_df['views'] * 100
yt_df['comment_rate'] =  yt_df ['comment_count'] / yt_df['views'] * 100
# Popular videos
vQ1 = yt_df.views.quantile(0.25)
vQ3 = yt_df.views.quantile(0.75)
lQ1 = yt_df.reaction_percent.quantile(0.25)
lQ3 = yt_df.reaction_percent.quantile(0.75)
vIQR = vQ3 - vQ1
lIQR = lQ3 - lQ1

# well_reacted_videos = yt_df.loc[yt_df.reaction_percent > (lQ3 + 1.5 * lIQR)]
# well_watched_videos = yt_df.loc[yt_df.views > (vQ3 + 1.5 * vIQR)]

yt_df['popular'] = 0
yt_df.loc[(yt_df.views > (vQ3 + 1.5 * vIQR)) & (yt_df.reaction_percent > (lQ1)),'popular'] = 1
popular_videos = yt_df[yt_df['popular'] == 1]
popular_videos.sort_values(by='like_dislike_ratio', ascending=False, inplace=True)
popular_videos.head(5)
# Helper function to count number of words in upper case
def numberOfUpper(string):
    i = 0
    for word in string.split():
        if word.isupper():
            i += 1
    return(i)

yt_df['all_upper_in_title'] = yt_df['title'].apply(numberOfUpper)

#Word count in each:
yt_df['count_word']=yt_df["title"].apply(lambda x: len(str(x).split()))
yt_df['count_word_tags']=yt_df["tags"].apply(lambda x: len(str(x).split()))

#Unique word count
yt_df['count_unique_word']=yt_df["title"].apply(lambda x: len(set(str(x).split())))
yt_df['count_unique_word_tags']=yt_df["tags"].apply(lambda x: len(set(str(x).split())))

#Letter count
yt_df['count_letters']=yt_df["title"].apply(lambda x: len(str(x)))
yt_df['count_letters_tags']=yt_df["tags"].apply(lambda x: len(str(x)))

#punctuation count
yt_df["count_punctuations"] =yt_df["title"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
yt_df["count_punctuations_tags"] =yt_df["tags"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count
yt_df["count_words_upper"] = yt_df["title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
yt_df["count_words_upper_tags"] = yt_df["tags"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count
yt_df["count_words_title"] = yt_df["title"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
yt_df["count_words_title_tags"] = yt_df["tags"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords
yt_df["count_stopwords"] = yt_df["title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
yt_df["count_stopwords_tags"] = yt_df["tags"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words
yt_df["mean_word_len"] = yt_df["title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
yt_df["mean_word_len_tags"] = yt_df["tags"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

#Word count percent in each:
yt_df['word_unique_percent']=yt_df['count_unique_word']*100/yt_df['count_word']
yt_df['word_unique_percent_tags']=yt_df['count_unique_word_tags']*100/yt_df['count_word_tags']

#Punct percent in each:
yt_df['punct_percent']=yt_df['count_punctuations']*100/yt_df['count_word']
yt_df['punct_percent_tags']=yt_df['count_punctuations_tags']*100/yt_df['count_word_tags']

yt_df.head(3)
f,ax = plt.subplots(figsize=(24, 17))
sns.heatmap(yt_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.title('Feature Correlations')
plt.show()
max_title_length = 20
number_of_videos = 20

viewers = yt_df.sort_values(["views"], ascending=False).head(number_of_videos)
viewers_title = [(x if len(x) <= max_title_length else x[:max_title_length] + "...") for x in viewers.title.values]
viewers_publish_hour = viewers.publish_hour.values
viewers_views = viewers.views.values

trace1 = go.Bar(
    x = viewers_title,
    y = viewers_publish_hour,
    name='Hour of Publishing',
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)
trace2 = go.Bar(
    x = viewers_title,
    y = viewers_views,
    name='Total Views',
    marker=dict(
        color='rgba(219, 64, 82, 0.7)',
        line=dict(
            color='rgba(219, 64, 82, 1.0)',
            width=2,
        )
    ),
    yaxis='y2'
)


data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title = 'Views',
    width=900,
    height=500,
    margin=go.Margin(
        l=75,
        r=75,
        b=120,
        t=80,
        pad=10
    ),
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        title= 'Hour of Publishing',
        anchor = 'x',
        rangemode='tozero'
    ),   
    yaxis2=dict(
        title='Total Number of Views',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 61000
    ),
    #legend=dict(x=-.1, y=1.2)
    legend=dict(x=0.1, y=0.05)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
max_title_length = 20
number_of_late_bloomers = 20

late_bloomers = yt_df.sort_values(["time_to_trend"], ascending=False).head(number_of_late_bloomers)
late_bloomers_title = [(x if len(x) <= max_title_length else x[:max_title_length] + "...") for x in late_bloomers.title.values]
late_bloomers_days = late_bloomers.time_to_trend.values
late_bloomers_views = late_bloomers.views.values

trace1 = go.Bar(
    x = late_bloomers_title,
    y = late_bloomers_days,
    name='Number of days',
    marker=dict(
        color='rgba(55, 128, 191, 0.7)',
        line=dict(
            color='rgba(55, 128, 191, 1.0)',
            width=2,
        )
    )
)
trace2 = go.Bar(
    x = late_bloomers_title,
    y = late_bloomers_views,
    name='total views',
    marker=dict(
        color='rgba(219, 64, 82, 0.7)',
        line=dict(
            color='rgba(219, 64, 82, 1.0)',
            width=2,
        )
    ),
    yaxis='y2'
)


data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title = 'Late bloomers',
    width=900,
    height=500,
    margin=go.Margin(
        l=75,
        r=75,
        b=120,
        t=80,
        pad=10
    ),
    paper_bgcolor='rgb(244, 238, 225)',
    plot_bgcolor='rgb(244, 238, 225)',
    yaxis = dict(
        title= 'Number of days until becoming trending',
        anchor = 'x',
        rangemode='tozero'
    ),   
    yaxis2=dict(
        title='Total number of views',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right',
        anchor = 'x',
        rangemode = 'tozero',
        dtick = 61000
    ),
    #legend=dict(x=-.1, y=1.2)
    legend=dict(x=0.1, y=0.05)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
plt.figure(figsize = (10,8))

sns.heatmap(yt_df[['like_rate', 'dislike_rate', 'comment_rate', 'comment_log',
         'views_log','likes_log','dislikes_log', 'category']].corr(), annot=True, linewidths=.5, fmt= '.1f')
plt.show()
plt.figure(figsize = (12,18))

plt.subplot(421)
g1 = sns.distplot(yt_df['count_word'], 
                  hist=False, label='Text')
g1 = sns.distplot(yt_df['count_word_tags'], 
                  hist=False, label='Tags')
g1.set_xlabel("Number of Words")
g1.set_title('Word Count Distribution', fontsize=16)

plt.subplot(422)
g2 = sns.distplot(yt_df['count_unique_word'],
                  hist=False, label='Text')
g2 = sns.distplot(yt_df['count_unique_word_tags'], 
                  hist=False, label='Tags')
g2.set_xlabel("Number of Unique Words")
g2.set_title('Unique Word Count Distribution', fontsize=16)

plt.subplot(423)
g3 = sns.distplot(yt_df['count_letters'], 
                  hist=False, label='Text')
g3 = sns.distplot(yt_df['count_letters_tags'], 
                  hist=False, label='Tags')
g3.set_xlabel("Number of Letters")
g3.set_title('Letter Count Distribution', fontsize=16)

plt.subplot(424)
g4 = sns.distplot(yt_df['count_punctuations'], 
                  hist=False, label='Text')
g4 = sns.distplot(yt_df['count_punctuations_tags'], 
                  hist=False, label='Tags')
g4.set_xlabel("Number of Punctuations")
g4.set_xlim([-2,50])
g4.set_title('Punctuation Count Distribution', fontsize=16)

plt.subplot(425)
g5 = sns.distplot(yt_df['count_words_upper'] , 
                  hist=False, label='Text')
g5 = sns.distplot(yt_df['count_words_upper_tags'] , 
                  hist=False, label='Tags')
g5.set_xlabel("Number of All Upper Case Words")
g5.set_title('Capitalized Word Count Distribution', fontsize=16)

plt.subplot(426)
g6 = sns.distplot(yt_df['count_words_title'], 
                  hist=False, label='Text')
g6 = sns.distplot(yt_df['count_words_title_tags'], 
                  hist=False, label='Tags')
g6.set_xlabel("Number of Title Case Words")
g6.set_title('Title Case Word Distribution', fontsize=16)

plt.subplot(427)
g7 = sns.distplot(yt_df['count_stopwords'], 
                  hist=False, label='Title')
# g7 = sns.distplot(yt_df['count_stopwords_tags'], 
#                   hist=False, label='Tags')
g7.set_xlabel("Number of Stopwords")
g7.set_title('Stopword Distribution', fontsize=16)

plt.subplot(428)
g8 = sns.distplot(yt_df['mean_word_len'], 
                  hist=False, label='Text')
g8 = sns.distplot(yt_df['mean_word_len_tags'], 
                  hist=False, label='Tags')
g8.set_xlabel("Mean Length of Words")
g8.set_xlim([-2,100])
g8.set_title('Mean Word Length Distribution', fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.legend()
plt.show()
plt.figure(figsize = (21,8))

plt.subplot(221)
g=sns.boxplot(x='count_punctuations', y='views_log',data=yt_df)
g.set_title("Views by Punctuations")
g.set_xlabel("Number of Punctuations")
g.set_ylabel("Views log")

plt.subplot(222)
g1 = sns.boxplot(x='count_punctuations', y='likes_log',data=yt_df)
g1.set_title("Likes by Punctuations")
g1.set_xlabel("Number of Punctuations")
g1.set_ylabel("Likes log")

plt.subplot(223)
g2 = sns.boxplot(x='count_punctuations', y='dislikes_log',data=yt_df)
g2.set_title("Dislikes by Punctuations")
g2.set_xlabel("Number of Punctuations")
g2.set_ylabel("Dislikes log")

plt.subplot(224)
g3 = sns.boxplot(x='count_punctuations', y='comment_log',data=yt_df)
g3.set_title("Comments by Punctuations")
g3.set_xlabel("Number of Punctuations")
g3.set_ylabel("Comments log")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)

plt.show()
plt.figure(figsize = (14,9))

plt.subplot(211)
cat_box = sns.boxplot(x='category', y='views_log', data=yt_df, palette="Set1")
cat_box.set_xticklabels(cat_box.get_xticklabels(),rotation=90)
cat_box.set_title("Views Distribution : Category", fontsize=15)
cat_box.set_xlabel("", fontsize=12)
cat_box.set_ylabel("Views(log)", fontsize=12)

plt.subplot(212)
count_box = sns.boxplot('category', 'reaction_percent', data=yt_df, palette="Set1")
count_box.set_xticklabels(count_box.get_xticklabels(),rotation=90)
count_box.set_title("Reaction Percent : Category", fontsize=15)
count_box.set_xlabel("", fontsize=12)
count_box.set_ylabel("Reaction %", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)

plt.show()
plt.figure(figsize = (30,9))

plt.subplot(221)
likes_cat = sns.boxplot(x='category', y='likes_log', data=yt_df, palette="Set1")
likes_cat.set_xticklabels(likes_cat.get_xticklabels(),rotation=90)
likes_cat.set_title("Likes Distribution : Category", fontsize=15)
likes_cat.set_xlabel("", fontsize=12)
likes_cat.set_ylabel("Likes(log)", fontsize=12)

plt.subplot(222)
dislikes_cat = sns.boxplot(x='category', y='dislikes_log', data=yt_df, palette="Set1")
dislikes_cat.set_xticklabels(dislikes_cat.get_xticklabels(),rotation=90)
dislikes_cat.set_title("Dislikes Distribution : Category", fontsize=15)
dislikes_cat.set_xlabel("", fontsize=12)
dislikes_cat.set_ylabel("Dislikes(log)", fontsize=12)

plt.subplot(223)
comment_cat = sns.boxplot(x='category', y='comment_log', data=yt_df, palette="Set1")
comment_cat.set_xticklabels(comment_cat.get_xticklabels(),rotation=90)
comment_cat.set_title("Comments Distribution : Category", fontsize=15)
comment_cat.set_xlabel("", fontsize=12)
comment_cat.set_ylabel("Comments Count(log)", fontsize=12)

plt.subplots_adjust(hspace = 0.9, top = 0.9)
plt.show()
plt.figure(figsize = (13,10))

sns.heatmap(yt_df[['count_word', 'count_unique_word','count_letters',
                     "count_punctuations","count_words_upper", "count_words_title", 
                     "count_stopwords","mean_word_len", 
                     'views_log', 'likes_log','dislikes_log','comment_log',
                     'ratings_disabled', 'comments_disabled', 'video_error_or_removed']].corr(), linewidths=0.1, fmt= '.1f', annot=True)
plt.show()
top_N = 100

# Get US/GB title and tags
yt_df.sort_values(by='views', ascending=False, inplace=True)
en_df = yt_df.loc[yt_df['country'].isin(['US', 'GB', 'CA'])]
desc_df = en_df.loc[:, en_df.columns.isin(['title', 'tags', 'description'])]
# Strip words from three features
raw_title_words = desc_df['title'].str.lower().str.cat(sep=' ')
raw_tags_words = desc_df['tags'].str.lower().str.cat(sep=' ')
raw_desc_words = desc_df['description'].str.lower().str.cat(sep=' ')


# wordcloud
def wc(data, bgcolor, title):
#     plt.figure(figsize = (15,15))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')

def popular_word_display(raw_words, pos, desc_type):
    # Removes punctuation,numbers and returns list of words
    desc_words = re.sub('[^A-Za-z]+', ' ', raw_words)

    word_tokens = word_tokenize(desc_words)
    filtered_sentence = [w for w in word_tokens if not w in eng_stopwords]
    filtered_sentence = []
    for w in word_tokens:
        if w not in eng_stopwords:
            filtered_sentence.append(w)

    # Remove characters which have length less than 2  
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]

    # Remove numbers
    cleaned_data = [word for word in without_single_chr if not word.isnumeric()]        

    # Calculate frequency distribution
    word_dist = nltk.FreqDist(cleaned_data)
    rslt = pd.DataFrame(word_dist.most_common(top_N),
                        columns=[desc_type, 'Frequency'])

    sns.set_style("whitegrid")
    ax = fig.add_subplot(gs[0, pos]) # row 0, col 0
    sns.barplot(x=desc_type,y="Frequency", data=rslt.head(7), ax=ax)
    ax1 = fig.add_subplot(gs[1,pos])
    wc(cleaned_data, 'black', 'Most Common Description Words' )
    
# Create 3x2 sub plots
fig = plt.figure(figsize=(27, 13), constrained_layout=False)
gs = fig.add_gridspec(ncols=3, nrows=2)

# Graph with wordcloud underneath
popular_word_display(raw_title_words, 0, 'Title Words')
popular_word_display(raw_tags_words, 1, 'Tag Words')
popular_word_display(raw_desc_words, 2, 'Description Words')
bloblist_title = list()

for row in desc_df['title']:
    blob = TextBlob(row)
    bloblist_title.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_polarity_title = pd.DataFrame(bloblist_title, columns = ['sentence','sentiment','polarity'])
def f_title(df_polarity_title):
    if df_polarity_title['sentiment'] > 0:
        val = "Positive"
    elif df_polarity_title['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val



df_polarity_title['Sentiment_Type'] = df_polarity_title.apply(f_title, axis=1)

plt.figure(figsize=(5,5))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_polarity_title)

pos_titles = df_polarity_title['sentence'][df_polarity_title['Sentiment_Type'] == 'Positive']
pos_title_words = pos_titles.str.lower().str.cat(sep=' ')

plt.figure(figsize=(13,9))
popular_word_display(pos_title_words, 0, 'Most Positive Title Video Words')
pos_title_words_list = ['live', 'win', 'love', 'new', 'best', 'good', 'real', 'special', 'perfect', 'right']
d_pos = en_df[en_df['title'].isin(pos_titles)].groupby('category').median().sort_values('views', ascending=False)
d_neg = en_df[~en_df['title'].isin(pos_titles)].groupby('category').median().sort_values('views', ascending=False)
plt.figure(figsize = (12,9))

plt.subplot(221)
g1 = sns.distplot(d_pos['views'],
                  hist=False, label='Positive')
g1 = sns.distplot(d_neg['views'],
                  hist=False, label='Non-Positive')
g1.set_title("Title Sentiment Effect on Views", fontsize=16)

plt.subplot(222)
g2 = sns.distplot(d_pos['likes'], 
                  hist=False, label='Positive')
g2 = sns.distplot(d_neg['likes'], 
                  hist=False, label='Non-Positive')
g2.set_title("Title Sentiment Effect on Likes", fontsize=16)

plt.subplot(223)
g3 = sns.distplot(d_pos['dislikes'], 
                  hist=False, label='Positive')
g3 = sns.distplot(d_neg['dislikes'], 
                  hist=False, label='Non-Positive')
g3.set_title("Title Sentiment Effect on Dislikes", fontsize=16)

plt.subplot(224)
g4 = sns.distplot(d_pos['comment_count'], 
                  hist=False, label='Positive')
g4 = sns.distplot(d_neg['comment_count'], 
                  hist=False, label='Non-Positive')
g4.set_title("Title Sentiment Effect on Comments", fontsize=16)

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.legend()
plt.show()
punctuations = string.punctuation
parser = English()

def spacy_tokenizer(sentence):
    # Removes punctuation,numbers and returns list of words
    desc_words = re.sub('[^A-Za-z]+', ' ', sentence)
    mytokens = parser(desc_words)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in eng_stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


tqdm.pandas()

normal = en_df["title"][en_df["popular"] == 0].progress_apply(spacy_tokenizer)
popular = en_df["title"][en_df["popular"] == 1].progress_apply(spacy_tokenizer)
#tokenize words by popularity 

def word_generator(text):
    word = list(text.split())
    return word
# def bigram_generator(text):
#     bgram = list(bigrams(text.split()))
#     bgram = [' '.join((a, b)) for (a, b) in bgram]
#     return bgram
# def trigram_generator(text):
#     tgram = list(trigrams(text.split()))
#     tgram = [' '.join((a, b, c)) for (a, b, c) in tgram]
#     return tgram


normal_words = normal.progress_apply(word_generator)
popular_words = popular.progress_apply(word_generator)
# normal_bigrams = normal.progress_apply(bigram_generator)
# popular_bigrams = popular.progress_apply(bigram_generator)
# normal_trigrams = normal.progress_apply(trigram_generator)
# popular_trigrams = popular.progress_apply(trigram_generator)
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(en_df.title)
word_features = word_vectorizer.transform(en_df.title)

classifier_popular = LogisticRegression(C=0.1, solver='sag')
classifier_popular.fit(word_features ,en_df.popular)
names=['normal','popular']
c_tf = make_pipeline(word_vectorizer, classifier_popular)
explainer_tf = LimeTextExplainer(class_names=names)

exp = explainer_tf.explain_instance(en_df.title.iloc[2], c_tf.predict_proba, num_features=4, top_labels=1)
exp.show_in_notebook(text=en_df.title.iloc[2])
exp = explainer_tf.explain_instance(en_df.title.iloc[9], c_tf.predict_proba, num_features=5, top_labels=1)
exp.show_in_notebook(text=en_df.title.iloc[9])
exp = explainer_tf.explain_instance(en_df.title.iloc[1100], c_tf.predict_proba, num_features=5, top_labels=1)
exp.show_in_notebook(text=en_df.title.iloc[1100])
exp = explainer_tf.explain_instance(en_df.title.iloc[18187], c_tf.predict_proba, num_features=5, top_labels=1)
exp.show_in_notebook(text=en_df.title.iloc[18187])
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(normal)
inp_sequences[:10]
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)
def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

model = create_model(max_sequence_len, total_words)
model.summary()
# model.fit(predictors, label, epochs=5, verbose=5)
model.fit(predictors, label, epochs=50, verbose=5)
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
def generate_title_texts(words_list):
    for word in words_list:
        print(generate_text(word, random.randint(4, 10)))
generate_title_texts(pos_title_words_list)
exp = explainer_tf.explain_instance('Best Worst Dressed Billboard Music Kills Today Laundry Fortnite Songs King', c_tf.predict_proba, num_features=5, top_labels=1)
exp.show_in_notebook(text='Best Worst Dressed Billboard Music Kills Today Laundry Fortnite Songs King')