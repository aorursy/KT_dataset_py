import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from gensim.summarization import keywords,summarize

from tqdm import tqdm

tqdm.pandas()

import matplotlib.pyplot as plt
alexa = pd.read_csv('../input/amazon_alexa.tsv', delimiter='\t')
alexa.head()
alexa.feedback.value_counts()
alexa.info()
pd.to_datetime(alexa['date']).min()
pd.to_datetime(alexa['date']).max()
#https://github.com/santosjorge/cufflinks/issues/185

!pip install plotly

!pip install cufflinks
import cufflinks as cf

cf.set_config_file(offline=True)
_=alexa.variation.value_counts().iplot(kind='bar')
_=alexa.rating.value_counts().iplot(kind='bar')
df = alexa.copy()
df['date'] = pd.to_datetime(df['date'])
pd.get_dummies(df[['date', 'rating']], columns=['rating']).head()
pd.crosstab(df['rating'], df['variation']).iplot(kind='bar', subplots=True)
df[['date', 'rating']].iplot(x='date', kind='scatter', mode='markers')
pd.get_dummies(df[['date', 'rating']], columns=['rating']).iplot(x='date', kind='bar', subplots=True)
df['date'].value_counts()[:5]
df_variation_dum = pd.get_dummies(df[['date', 'variation']], columns=['variation'])
df_variation_dum.iplot(x='date', kind='bar', barmode='stack')
df_variation_dum.iplot(x='date', kind='bar', subplots=True)
#https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings

df_key = df.groupby('date')['verified_reviews'].agg(lambda rev: ' '.join(rev)).progress_apply(lambda review: keywords(review, 

                                                                                    scores=True, lemmatize= True, split=True))
df_key[:5].progress_apply(lambda key : len(key)) 
df_key['len_key'] = df_key.progress_apply(lambda key : len(key)) 
#df_key['len_key'].value_counts()
#alexa['keywords'] =  alexa['verified_reviews'].progress_apply(lambda review: keywords(review))
 #https://stackoverflow.com/questions/27298178/concatenate-strings-from-several-rows-using-pandas-groupby

#alexa_rating_keywords = alexa[['verified_reviews','rating']].groupby(['rating'])['verified_reviews'].apply(list).apply('. '.join).apply(lambda review: keywords(review, ratio=0.01)).reset_index()
#alexa_rating_keywords
alexa_rating_summary = alexa[['verified_reviews','rating']].groupby(['rating'])['verified_reviews'].apply(list).apply('. '.join).apply(lambda review: summarize(review, word_count=100)).reset_index()

#ratio=0.05
for i in range(5):

    print(alexa_rating_summary.iloc[i])
alexa_rating_summary = alexa[['verified_reviews','rating']].groupby(['rating'])['verified_reviews'].apply(list).apply('. '.join).apply(lambda review: summarize(review, word_count=100)).reset_index()

alexa_rating_keywords = alexa[['verified_reviews','rating']].groupby(['rating'])['verified_reviews'].apply(list).apply('. '.join).apply(lambda review: keywords(review, split=True,scores=True,lemmatize=True)).reset_index()
alexa_rating_keywords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

alexa_rating_sentiment = alexa[['verified_reviews','rating']].groupby(['rating'])['verified_reviews'].apply(list).apply('. '.join).apply(lambda review: analyser.polarity_scores(review)).reset_index()
alexa_rating_sentiment
df = pd.concat([alexa_rating_keywords.rating,alexa_rating_keywords.verified_reviews,alexa_rating_summary.verified_reviews,alexa_rating_sentiment.verified_reviews],axis=1)
df.columns =  ['rating','keywords','summary','sentiment']
print(df[df.rating==1].iloc[0,1])
print(df[df.rating==1].iloc[0,2])
print(df[df.rating==2].iloc[0,1])
print(df[df.rating==2].iloc[0,2])
print(df[df.rating==3].iloc[0,1])
print(df[df.rating==3].iloc[0,2])
print(df[df.rating==4].iloc[0,1])
print(df[df.rating==4].iloc[0,2])
print(df[df.rating==5].iloc[0,1])
print(df[df.rating==5].iloc[0,2])
pd.set_option("display.max_colwidth",-1)

print(df.iloc[:,3])
alexa['len'] = alexa['verified_reviews'].str.len()
'''import seaborn as sns

sns.set()

sns.set(rc={'figure.figsize':(15,25)})

g = sns.catplot(x="len", y="date", hue="rating", data=alexa, height=10, aspect=13/10, orient='h')

g.set_xticklabels(rotation=90)

'''
import seaborn as sns

sns.set()

sns.set(rc={'figure.figsize':(10,10)})

_=alexa.pivot_table(index='variation',columns='rating',values='verified_reviews', aggfunc='count').plot(kind='barh')
#https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

#https://stackoverflow.com/questions/3292643/python-convert-list-of-tuples-to-string

import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()

#nlp = spacy.load('en',tagger=False,parser=False,matcher=False)

def ent_names_text(review):

    doc = nlp(review)

    ls = []

    for X in doc.ents:

        if (X.text.strip()!=''):

            ls.append(X.text + '-' +  X.label_)

    

    return str(Counter(ls).most_common(4)).strip('[]')  

alexa_rating_entity = alexa[['verified_reviews','rating']].groupby(['rating'])['verified_reviews'].apply(list).apply('. '.join).progress_apply(lambda review: ent_names_text(review)).reset_index()
alexa_rating_entity
sns.set(rc={'figure.figsize':(10,10)})

al_count = alexa[['verified_reviews','rating']].groupby('rating').agg(['count'])

al_count.columns = al_count.columns.droplevel()

al_count.plot(kind='barh')
import seaborn as sns

sns.set()

sns.set(rc={'figure.figsize':(15,25)})

ax = sns.scatterplot(x="len", y="date", data=alexa)