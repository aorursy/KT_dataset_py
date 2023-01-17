# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install NRCLex
import matplotlib.pyplot as plt

import cufflinks as cf

import plotly

import plotly.express as px

import numpy as np

import re

import nltk

import plotly.graph_objs as go

import random

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg



from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,

                                  AnnotationBbox)

from matplotlib.cbook import get_sample_data

from wordcloud import WordCloud,STOPWORDS

from sklearn.feature_extraction.text import CountVectorizer

from pandas import DataFrame

from PIL import Image

from plotly import tools

from plotly.offline import init_notebook_mode,iplot,plot

from nrclex import NRCLex

from nltk.corpus import stopwords

stop = stopwords.words('english')
cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
df = pd.read_csv('../input/the-simpsons-dataset/simpsons_script_lines.csv')

df.head(5)
df = df.head(6000)
df.shape
df = df.dropna()
df['word_count'] = df['word_count'].astype(str).astype(int)
df.dtypes
description_list=[]

for description in df['normalized_text']:

    description=re.sub("[^a-zA-Z]", " ", description)

    description=description.lower()

    description=nltk.word_tokenize(description)

    description=[word for word in description if not word in set(stopwords.words("english"))]

    lemma=nltk.WordNetLemmatizer()

    description=[lemma.lemmatize(word) for word in description]

    description=" ".join(description)

    description_list.append(description)

df["normalized_text_new"]=description_list

df.head(5)
for i,row in df.iterrows():

    print(row['character_id'],row['raw_character_text'])
val_homer=[]

val_bart=[]

val_marge=[]

val_lisa=[]



for i,row in df.iterrows():

    val = row['normalized_text_new']

    if row['character_id'] == 2:

        val_homer.append(val)

    elif row['character_id']== 8:

        val_bart.append(val)

    elif row['character_id'] == 1:

        val_marge.append(val)

    elif row['character_id']== 9:

        val_lisa.append(val)    

pat = r'\b(?:{})\b'.format('|'.join(stop))

def text_cleaning(val_list):

    df1 = DataFrame (val_list,columns =['normalized_text_new']).dropna()

    df1["normalized_text_new"] = df1["normalized_text_new"].str.replace(pat, '')

    df1["normalized_text_new"] = df1["normalized_text_new"].str.replace(r'\s+', ' ')

    return df1
bart = text_cleaning(val_bart)

homer = text_cleaning(val_homer)

marge = text_cleaning(val_marge)

lisa = text_cleaning(val_lisa)
wc = WordCloud()

wc.generate(' '.join(bart['normalized_text_new']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
wc = WordCloud(background_color="white", max_words=2000,

               stopwords=STOPWORDS, max_font_size=256,

               random_state=42, width=500, height=500)

wc.generate(' '.join(bart['normalized_text_new']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
plt.subplots(figsize=(10,10))

mask = np.array(Image.open('../input/intermediate-notebooks-data/bart.png'))

wc = WordCloud(stopwords=STOPWORDS, 

               mask=mask, background_color="white", contour_width=2, contour_color="orange",

               max_words=2000, max_font_size=256,

               random_state=42, width=mask.shape[1],

               height=mask.shape[0])

wc.generate(' '.join(bart['normalized_text_new']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]





def get_top_n_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
common_words = get_top_n_words(bart['normalized_text_new'], 20)

common_words_df = DataFrame (common_words,columns=['word','freq'])

character_img = mpimg.imread('../input/intermediate-notebooks-data/bart1.jpg')

imagebox = OffsetImage(character_img, zoom=0.07)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy,

                    xybox=(17, 37),

                    pad=0.5,frameon=False

                    )

plt.figure(figsize=(16, 6))

ax = sns.barplot(x='word', y='freq', data=common_words_df,palette='hls')

ax.add_artist(ab)



plt.title("Top 20 unigrams used by Bart", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
common_words = get_top_n_bigram(bart['normalized_text_new'], 20)

df3 = pd.DataFrame(common_words, columns = ['words' ,'count'])

df3.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams used by Bart')



common_words = get_top_n_trigram(bart['normalized_text_new'], 20)

df4 = pd.DataFrame(common_words, columns = ['words' , 'count'])

df4.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams used by Bart')
plt.subplots(figsize=(10,10))

mask = np.array(Image.open('../input/intermediate-notebooks-data/lisa.jpg'))

wc = WordCloud(stopwords=STOPWORDS, 

               mask=mask, background_color="white", contour_width=2, contour_color="orange",

               max_words=2000, max_font_size=256,

               random_state=42, width=mask.shape[1],

               height=mask.shape[0])

wc.generate(' '.join(lisa['normalized_text_new']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
common_words = get_top_n_words(lisa['normalized_text_new'], 20)

common_words_df = DataFrame (common_words,columns=['word','freq'])

character_img = mpimg.imread('../input/intermediate-notebooks-data/lisa1.jpg')

imagebox = OffsetImage(character_img, zoom=0.4)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy,

                    xybox=(16, 30),

                    pad=0.5,frameon=False

                    )

plt.figure(figsize=(16, 6))

ax = sns.barplot(x='word', y='freq', data=common_words_df,palette='hls')

ax.add_artist(ab)



plt.title("Top 20 unigrams used by Lisa", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
common_words = get_top_n_bigram(lisa['normalized_text_new'], 20)

df3 = pd.DataFrame(common_words, columns = ['words' ,'count'])

df3.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams used by Lisa')



common_words = get_top_n_trigram(lisa['normalized_text_new'], 20)

df4 = pd.DataFrame(common_words, columns = ['words' , 'count'])

df4.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams used by Lisa')
plt.subplots(figsize=(10,10))

mask = np.array(Image.open('../input/intermediate-notebooks-data/homer.jpg'))

wc = WordCloud(stopwords=STOPWORDS, 

               mask=mask, background_color="white", contour_width=2, contour_color="black",

               max_words=2000, max_font_size=256,

               random_state=42, width=mask.shape[1],

               height=mask.shape[0])

wc.generate(' '.join(homer['normalized_text_new']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
common_words = get_top_n_words(homer['normalized_text_new'], 20)

common_words_df = DataFrame (common_words,columns=['word','freq'])

character_img = mpimg.imread('../input/intermediate-notebooks-data/homer1.jpg')

imagebox = OffsetImage(character_img, zoom=0.2)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy,

                    xybox=(17, 78),

                    pad=0.5,frameon=False

                    )

plt.figure(figsize=(16, 7))

ax = sns.barplot(x='word', y='freq', data=common_words_df,palette='hls')

ax.add_artist(ab)



plt.title("Top 20 unigrams used by Homer", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
common_words = get_top_n_bigram(homer['normalized_text_new'], 20)

df3 = pd.DataFrame(common_words, columns = ['words' ,'count'])

df3.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams used by Homer')



common_words = get_top_n_trigram(homer['normalized_text_new'], 20)

df4 = pd.DataFrame(common_words, columns = ['words' , 'count'])

df4.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams used by Homer')
plt.subplots(figsize=(10,10))

mask = np.array(Image.open('../input/intermediate-notebooks-data/marge.png'))

wc = WordCloud(stopwords=STOPWORDS, 

               mask=mask, background_color="white", contour_width=2, contour_color="green",

               max_words=2000, max_font_size=256,

               random_state=42, width=mask.shape[1],

               height=mask.shape[0])

wc.generate(' '.join(marge['normalized_text_new']))

plt.imshow(wc, interpolation="bilinear")

plt.axis('off')

plt.show()
common_words = get_top_n_words(marge['normalized_text_new'], 20)

common_words_df = DataFrame (common_words,columns=['word','freq'])



character_img = mpimg.imread('../input/intermediate-notebooks-data/marge1.jpg')

imagebox = OffsetImage(character_img, zoom=0.18)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy,

                    xybox=(17, 65),

                    pad=0.5,frameon=False

                    )

plt.figure(figsize=(16, 6))

ax = sns.barplot(x='word', y='freq', data=common_words_df,palette='hls')

ax.add_artist(ab)



plt.title("Top 20 unigrams used by Marge", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
common_words = get_top_n_bigram(marge['normalized_text_new'], 20)

df3 = pd.DataFrame(common_words, columns = ['words' ,'count'])

df3.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams used by Marge')



common_words = get_top_n_trigram(marge['normalized_text_new'], 20)

df4 = pd.DataFrame(common_words, columns = ['words' , 'count'])

df4.groupby('words').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams used by Marge')
text_object = NRCLex(' '.join(df['normalized_text_new']))
text_object.affect_frequencies
text_object.top_emotions
sentiment_scores = pd.DataFrame(list(text_object.raw_emotion_scores.items()),columns = ['sentiment','scores']) 
fig = px.pie(sentiment_scores, values='scores', names='sentiment',

             title='Sentiment Scores',

             hover_data=['sentiment'], labels={'sentiment':'sentiment'})

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
sentiment_words = pd.DataFrame(list(text_object.affect_dict.items()),columns = ['words','sentiments']).tail(5)

sentiment_words