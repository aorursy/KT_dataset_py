# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.cloud import bigquery

from google.cloud import storage



from keras.preprocessing.text import Tokenizer

from nltk.tokenize import word_tokenize

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm



from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

import spacy 

from collections import Counter

import re

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix

from datetime import datetime



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Set your own project id here

PROJECT_ID = 'covid19nycnews'





from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import automl_v1beta1 as automl

#automl_client = automl.AutoMlClient()
PROJECT_ID = 'covid19nycnews'



# Import the BQ API Client library

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID, location='US')
import os



print('Credendtials from environ: {}'.format(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))


# Construct a reference to the Ames Housing dataset that is within the project

dataset_ref = client.dataset('nycnewsdataset', project=PROJECT_ID)



# Make an API request to fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Make a list of all the tables in the dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset

for table in tables:  

    print(table.table_id)
# Preview the first five lines of the table

client.list_rows(table, max_results=5).to_dataframe()
df= client.list_rows(table,max_results=300000).to_dataframe()
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

df.Date.dt.month

df.groupby(df['Date'].dt.month).size()
import spacy
# nlp = spacy.load('en')



# feats = ['char_count', 'word_count', 'word_count_cln',

#        'stopword_count']



# class text_features:

#     def __init__(self, df, textcol):

#         self.df = df

#         self.textcol = textcol

#         self.c = "spacy_" + textcol

#         self.df[self.c] = self.df[self.textcol].apply( lambda x : nlp(x))

        

#     def _spacy_cleaning(self, doc):

#         tokens = [token for token in doc if (token.is_stop == False)\

#                   and (token.is_punct == False)]

#         words = [token.lemma_ for token in tokens]

#         return " ".join(words)

        

#     def _spacy_features(self):

#         self.df["clean_text"] = self.df[self.c].apply(lambda x : self._spacy_cleaning(x))

#         self.df["char_count"] = self.df[self.textcol].apply(len)

#         self.df["word_count"] = self.df[self.c].apply(lambda x : len([_ for _ in x]))

#         self.df["word_count_cln"] = self.df["clean_text"].apply(lambda x : len(x.split()))

        

#         self.df["stopword_count"] = self.df[self.c].apply(lambda x : 

#                                                           len([_ for _ in x if _.is_stop]))

        

                

#     def generate_features(self):

#         self._spacy_features()

#         self.df = self.df.drop([self.c,"clean_text"], axis=1)

#         return self.df

    

    

# def spacy_features(df, tc):

#     fe = text_features(df, tc)

#     return fe.generate_features()
# textcol = "ContextualText"

# text_features = spacy_features(df, textcol)

# text_features[[textcol] + feats].head()
##Creating a list of stop words and adding custom stopwords

stop_words = set(stopwords.words("english"))

corpus= df['ContextualText']

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))

X=cv.fit_transform(corpus)





 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(X)

# get feature names

feature_names=cv.get_feature_names()

 

# fetch document for which keywords needs to be extracted

doc=corpus[1]

 

#generate tf-idf for the given document

tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))



#Function for sorting tf_idf in descending order



def sort_coo(coo_matrix):

    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

 

def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    

    #use only topn items from vector

    sorted_items = sorted_items[:topn]

 

    score_vals = []

    feature_vals = []

    

    # word index and corresponding tf-idf score

    for idx, score in sorted_items:

        

        #keep track of feature name and its corresponding score

        score_vals.append(round(score, 3))

        feature_vals.append(feature_names[idx])

 

    #create a tuples of feature,score

    #results = zip(feature_vals,score_vals)

    results= {}

    for idx in range(len(feature_vals)):

        results[feature_vals[idx]]=score_vals[idx]

    

    return results

#sort the tf-idf vectors by descending order of scores

sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10

keywords=extract_topn_from_vector(feature_names,sorted_items,5)

 

# now print the results

print("\nAbstract:")

print(doc)

print("\nKeywords:")

for k in keywords:

    print(k,keywords[k])
keywords_arr=[]

for i in range(len(df['ContextualText'])):

# fetch document for which keywords needs to be extracted

    doc=corpus[i]



    #generate tf-idf for the given document

    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

    

    #sort the tf-idf vectors by descending order of scores

    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10

    keywords=extract_topn_from_vector(feature_names,sorted_items,5)

    print("\nAbstract:")

    print(doc)

    top_5= list(keywords.keys())

    print(i,"keywords:",top_5)

    keywords_arr.append(top_5)
pd.DataFrame(keywords_arr)
df.to_csv('sample_data.csv', index=False)