# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import spacy 

nlp = spacy.load('en', parse=True, tag=True, entity=True)

import nltk

from nltk.tokenize.toktok import ToktokTokenizer

import sqlite3

import os

import re

tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')

stopword_list.remove('no')

stopword_list.remove('not')

import unicodedata



# Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



sql_conn = sqlite3.connect('../input/database.sqlite')

corpus = pd.read_sql("SELECT author, body, ups, downs, controversiality,removal_reason,score,score_hidden FROM May2015 where subreddit = 'psychology' and LENGTH(body) > 30 AND LENGTH(body) < 250 LIMIT 10000", sql_conn)



corpus.to_csv('psychology.csv', index=False)

corpus.head()
## Step 1: Text Preprocessing

def remove_special_char(text):

    #replace special characters with ''

    text = re.sub('[^\w\s]', '', text)

    #remove chinese character

    text = re.sub(r'[^\x00-\x7f]',r'', text)

    #remove numbers

    text = re.sub('\d+', '', text)

    text = re.sub('_', '', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip()

    #text = text.lower()

    #remove accented characters

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text

    

def remove_stopwords(text):

    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    text = [token for token in tokens if token.lower() not in stopword_list]

    return " ".join(text)



def stem_text(text):

    stemmer = nltk.porter.PorterStemmer()

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text)



def lemmatize_text(text):

    text = nlp(text)

    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    return text

def normalize_corpus(corpus):

    

    normalized_corpus = []

    # normalize each document in the corpus

    for doc in corpus:

        # remove special character and normalize docs

        doc = remove_special_char(doc)

        # remove stopwards 

        doc = remove_stopwords(doc)

        # lemmatize docs    

        doc = lemmatize_text(doc)

        normalized_corpus.append(doc)

        

    return normalized_corpus
normalized_corpus = normalize_corpus(corpus['body'])

normalized_corpus[0]
from textblob import TextBlob



# compute sentiment scores (polarity) and labels

sentiment_scores_tb = [round(TextBlob(article).sentiment.polarity, 3) for article in normalized_corpus]

sentiment_category_tb = ['positive' if score > 0 

                             else 'negative' if score < 0 

                                 else 'neutral' 

                                     for score in sentiment_scores_tb]





# sentiment statistics per news category

df_revised = pd.DataFrame([normalized_corpus, sentiment_scores_tb, sentiment_category_tb]).T

df_revised.columns = ['body', 'sentiment_score', 'sentiment_category']

df_revised['sentiment_score'] = df_revised.sentiment_score.astype('float')



df_revised.head()
sentiment_spread = (df_revised.groupby(by=['sentiment_category'])

                           .size()

                           .reset_index().rename(columns={0 : 'Frequency'}))

print(sentiment_spread)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True) #do not miss this line



data_bar = [go.Bar(

            x=sentiment_spread['sentiment_category'],

            y=sentiment_spread['Frequency']

    )]

layout = go.Layout(

    autosize=False,

    width=500,

    height=500

)



fig = go.Figure(data=data_bar, layout = layout)

py.offline.iplot(fig, filename='basic-bar')
from spacy import displacy



df_revised['entity_type'] = None

df_revised['named_entity'] = None



def ner_tagging(corpus):

    for doc in corpus['body']:

        temp_entity_name=''

        temp_named_entity =''

        sentence_nlp = nlp(doc)

        for word in sentence_nlp:

            if word.ent_type_:

                temp_entity_name = ' '.join([temp_entity_name,word.ent_type_]).strip()

                temp_named_entity = ' '.join([temp_named_entity,word.text]).strip()

        corpus.loc[corpus['body']== doc,['entity_type']]=temp_entity_name

        corpus.loc[corpus['body']== doc,['named_entity']]=temp_named_entity

    return corpus



# print named entities in article

#print([(word, word.ent_type_) for word in sentence_nlp if word.ent_type_])



# visualize named entities

#displacy.render(sentence_nlp, style='ent', jupyter=True)

df_with_NER = ner_tagging(df_revised)

df_with_NER.head()
named_entity_positive = []

named_entity_negative = []

named_entity_neutral = []



for index, row in df_with_NER.iterrows():

    temp = row['named_entity'].split()

    if row['sentiment_category']=='positive':

        named_entity_positive.extend(temp)

    if row['sentiment_category']=='negative':

        named_entity_negative.extend(temp)

    if row['sentiment_category']=='neutral':

        named_entity_neutral.extend(temp)



temp_pos = pd.DataFrame([named_entity_positive, ['positive']*len(named_entity_positive)]).T

temp_neg = pd.DataFrame([named_entity_negative, ['negative']*len(named_entity_negative)]).T

temp_neu = pd.DataFrame([named_entity_neutral, ['neutral']*len(named_entity_neutral)]).T



named_entity_df = pd.concat([temp_pos,temp_neg,temp_neu])

named_entity_df.columns = ['entity', 'sentiment']
named_entity_df.head()
entity_type_positive = []

entity_type_negative = []

entity_type_neutral = []



for index, row in df_with_NER.iterrows():

    temp = row['entity_type'].split()

    if row['sentiment_category']=='positive':

        entity_type_positive.extend(temp)

    if row['sentiment_category']=='negative':

        entity_type_negative.extend(temp)

    if row['sentiment_category']=='neutral':

        entity_type_neutral.extend(temp)



temp_pos = pd.DataFrame([entity_type_positive, ['positive']*len(entity_type_positive)]).T

temp_neg = pd.DataFrame([entity_type_negative, ['negative']*len(entity_type_negative)]).T

temp_neu = pd.DataFrame([entity_type_neutral, ['neutral']*len(entity_type_neutral)]).T



entity_type_df = pd.concat([temp_pos,temp_neg,temp_neu])

entity_type_df.columns = ['type', 'sentiment']

entity_type_df.head()
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

from wordcloud import WordCloud

% matplotlib inline



def display_wordcloud(data):

    d = {}

    for a, x in data.values:

        d[a] = x

    wordcloud=WordCloud(background_color="white").generate_from_frequencies(frequencies=d)

    return wordcloud



fig = plt.figure(figsize=(30,20))

i=0

for item in named_entity_df.sentiment.unique():

    ax = fig.add_subplot(1,3,i+1)

    temp = named_entity_df[named_entity_df['sentiment']==item]['entity']

    temp_freq=temp.value_counts()

    data_freq = pd.DataFrame({'entity':temp_freq.index, 'Frequency':temp_freq.values})

    wordcloud = display_wordcloud(data_freq)

    ax.imshow(wordcloud)

    ax.axis('off')

    i=i+1

plt.show() 





type_freq= (entity_type_df.groupby(by=['type']).size().reset_index().sort_values(0, ascending=False).rename(columns={0 : 'Frequency'}))

x_pos = type_freq['Frequency']

y_pos = type_freq['type']



fig_bar, ax = plt.subplots()

ax.barh(y_pos,x_pos, color='green')

plt.ylabel("Entity Type")

plt.xlabel("Frequency")

plt.title("Frequency of Entity Types")

plt.show()




