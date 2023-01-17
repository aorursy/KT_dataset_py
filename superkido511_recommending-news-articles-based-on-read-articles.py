import numpy as np

import datetime

import pandas as pd

import os

import math

import time



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.figure_factory as ff

import plotly.graph_objects as go

import plotly.express as px



# Below libraries are for text processing using NLTK

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer



# Below libraries are for feature representation using sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# Below libraries are for similarity matrices using sklearn

from sklearn.metrics.pairwise import cosine_similarity  

from sklearn.metrics import pairwise_distances
import json

with open('/kaggle/input/news-simple/FinalRelatedDataSimple.json', encoding='utf-8') as json_file:

    data = json.load(json_file)
news_list = []

date_format = '%Y-%m-%dT%H:%M:%S.%f%z'
for d in data:

    if "related" in d:

        for r in d["related"]:

            if r not in news_list:

                news_list.append(r)

        news_list.append(d['page'])
news_articles = pd.read_json("/kaggle/input/hoang-data/data.json", lines = False)
news_articles.info()
# news_articles["datetime_str"] = news_articles["published_at"].map(lambda a: str(a))

news_articles.head()
news_articles = news_articles.sort_values(by='published_at', ascending=False)
news_articles.head()
news_articles.shape
news_articles = news_articles[news_articles['name'].apply(lambda x: len(x.split())>5)]

print("Total number of articles after removal of headlines with short title:", news_articles.shape[0])
news_articles.sort_values('name',inplace=True, ascending=False)

duplicated_articles_series = news_articles.duplicated('name', keep = False)

news_articles = news_articles[~duplicated_articles_series]

print("Total number of articles after removing duplicates:", news_articles.shape[0])
news_articles.isna().sum()
fig = go.Figure([go.Bar(x=news_articles["category"].value_counts().index, y=news_articles["category"].value_counts().values)])

fig['layout'].update(title={"text" : 'Distribution of articles category-wise','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Category name",yaxis_title="Number of articles")

fig.update_layout(width=800,height=700)

fig
fig = go.Figure([go.Bar(x=news_articles_per_month.index.strftime("%b"), y=news_articles_per_month)])

fig['layout'].update(title={"text" : 'Distribution of articles month-wise','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Month",yaxis_title="Number of articles")

fig.update_layout(width=500,height=500)

fig
fig = ff.create_distplot([news_articles['name'].str.len()], ["ht"],show_hist=False,show_rug=False)

fig['layout'].update(title={'text':'PDF','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Length of a headline",yaxis_title="probability")

fig.update_layout(showlegend = False,width=500,height=500)

fig
news_articles.index = range(news_articles.shape[0])
news_articles.head()
news_articles_temp = news_articles.copy()
stop_words = set(stopwords.words('english'))
stop_words = open("/kaggle/input/vietnamesestopwords/vietnamese-stopwords.txt", "r", encoding='utf8').readlines()
stop_words = set([x.strip('\n') for x in stop_words])
for i in range(len(news_articles_temp["name"])):

    string = ""

    for word in news_articles_temp["name"][i].split():

        word = ("".join(e for e in word if e.isalnum()))

        word = word.lower()

        if not word in stop_words:

            string += word + " "  

    if(i%100==0):

        print(i)           # To track number of records processed

    news_articles_temp.at[i,"name"] = string.strip()
lemmatizer = WordNetLemmatizer()
for i in range(len(news_articles_temp["name"])):

    string = ""

    for w in word_tokenize(news_articles_temp["name"][i]):

        string += lemmatizer.lemmatize(w,pos = "v") + " "

    news_articles_temp.at[i, "name"] = string.strip()

    if(i%100==0):

        print(i)           # To track number of records processed
headline_vectorizer = CountVectorizer()

headline_features   = headline_vectorizer.fit_transform(news_articles_temp['name'])
news_articles_temp.head()

headline_features.get_shape()
pd.set_option('display.max_colwidth', -1)  # To display a very long headline completely
def bag_of_words_based_model(row_index, num_similar_items):

    couple_dist = pairwise_distances(headline_features,headline_features[row_index])

    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]

    df = pd.DataFrame({'publish_date': news_articles['published_at'][indices].values,

               'headline':news_articles['name'][indices].values,

                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})

    print("="*30,"Queried article details","="*30)

    print('headline : ',news_articles['name'][indices[0]])

    print("\n","="*25,"Recommended articles : ","="*23)

    #return df.iloc[1:,1]

    return df.iloc[1:,]



bag_of_words_based_model(133, 11) # Change the row index for any other queried article
tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)

tfidf_headline_features = tfidf_headline_vectorizer.fit_transform(news_articles_temp['name'])
def tfidf_based_model(row_index, num_similar_items):

    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])

    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]

    df = pd.DataFrame({'publish_date': news_articles['published_at'][indices].values,

               'headline':news_articles['name'][indices].values,

                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})

    print("="*30,"Queried article details","="*30)

    print('headline : ',news_articles['name'][indices[0]])

    print("\n","="*25,"Recommended articles : ","="*23)

    

    #return df.iloc[1:,1]

    return df.iloc[1:,]

tfidf_based_model(133, 11)
from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle
!ls
loaded_model = KeyedVectors.load_word2vec_format('/kaggle/input/popular-embedding/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
vocabulary = loaded_model.vocab.keys()

w2v_headline = []

for i in news_articles_temp["name"]:

    w2Vec_word = np.zeros(300, dtype="float32")

    for word in i.split():

        if word in vocabulary:

            w2Vec_word = np.add(w2Vec_word, loaded_model[word])

    w2Vec_word = np.divide(w2Vec_word, len(i.split()))

    w2v_headline.append(w2Vec_word)

w2v_headline = np.array(w2v_headline)
def avg_w2v_based_model(row_index, num_similar_items):

    couple_dist = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))

    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]

    df = pd.DataFrame({'publish_date': news_articles['published_at'][indices].values,

               'headline':news_articles['name'][indices].values,

                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})

    print("="*30,"Queried article details","="*30)

    print('headline : ',news_articles['name'][indices[0]])

    print("\n","="*25,"Recommended articles : ","="*23)

    #return df.iloc[1:,1]

    return df.iloc[1:,]



avg_w2v_based_model(133, 11)
from sklearn.preprocessing import OneHotEncoder 
category_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_articles_temp["category"]).reshape(-1,1))
def avg_w2v_with_category(row_index, num_similar_items, w1,w2): #headline_preference = True, category_preference = False):

    w2v_dist  = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))

    category_dist = pairwise_distances(category_onehot_encoded, category_onehot_encoded[row_index]) + 1

    weighted_couple_dist   = (w1 * w2v_dist +  w2 * category_dist)/float(w1 + w2)

    indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()

    df = pd.DataFrame({'publish_date': news_articles['published_at'][indices].values,

               'headline':news_articles['name'][indices].values,

                'Weighted Euclidean similarity with the queried article': weighted_couple_dist[indices].ravel(),

                'Word2Vec based Euclidean similarity': w2v_dist[indices].ravel(),

                 'Category based Euclidean similarity': category_dist[indices].ravel(),

                'Categoty': news_articles['category'][indices].values})

    

    print("="*30,"Queried article details","="*30)

    print('headline : ',news_articles['name'][indices[0]])

    print('Categoty : ', news_articles['category'][indices[0]])

    print("\n","="*25,"Recommended articles : ","="*23)

    #return df.iloc[1:,[1,5]]

    return df.iloc[1:, ]



avg_w2v_with_category(528,10,0.1,0.8)
publishingday_onehot_encoded = OneHotEncoder().fit_transform(np.array(news_articles_temp["published_at"]).reshape(-1,1))
def avg_w2v_with_category_and_publshing_day(row_index, num_similar_items, w1,w2,w3): #headline_preference = True, category_preference = False):

    w2v_dist  = pairwise_distances(w2v_headline, w2v_headline[row_index].reshape(1,-1))

    category_dist = pairwise_distances(category_onehot_encoded, category_onehot_encoded[row_index]) + 1

    publishingday_dist = pairwise_distances(publishingday_onehot_encoded, publishingday_onehot_encoded[row_index]) + 1

    weighted_couple_dist   = (w1 * w2v_dist +  w2 * category_dist + w3 * publishingday_dist)/float(w1 + w2 + w3)

    indices = np.argsort(weighted_couple_dist.flatten())[0:num_similar_items].tolist()

    df = pd.DataFrame({'publish_date': news_articles['published_at'][indices].values,

                'headline_text':news_articles['name'][indices].values,

                'Weighted Euclidean similarity with the queried article': weighted_couple_dist[indices].ravel(),

                'Word2Vec based Euclidean similarity': w2v_dist[indices].ravel(),

                'Category based Euclidean similarity': category_dist[indices].ravel(),  

                'Publishing day based Euclidean similarity': publishingday_dist[indices].ravel(), 

                'Categoty': news_articles['category'][indices].values})

    print("="*30,"Queried article details","="*30)

    print('headline : ',news_articles['name'][indices[0]])

    print('Categoty : ', news_articles['category'][indices[0]])

    print('Published at : ', news_articles['published_at'][indices[0]])

    print("\n","="*25,"Recommended articles : ","="*23)

    #return df.iloc[1:,[1,7,8,9]]

    # df = df.sort_values(by='publish_date', ascending=False)

    return df.iloc[1:, ]





avg_w2v_with_category_and_publshing_day(528,10,0.1,0.1,1)