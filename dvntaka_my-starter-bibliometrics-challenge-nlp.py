# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pickle

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from textblob import TextBlob

from collections import Counter

from nltk.util import ngrams



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation,  TruncatedSVD 



#nltk.download('stopwords')

#nltk.download('punkt')



#For generating word clouds from: https://www.geeksforgeeks.org/generating-word-cloud-python/

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 



import os

#raw_data = pd.read_csv('/kaggle/input/alsa/pubmed_als_articles.csv')



# we are going to read in the data and parse the date column as dates

raw_data = pd.read_csv('/kaggle/input/alsa/pubmed_als_articles.csv', encoding='utf8', parse_dates=['publication_date'],index_col=None)



raw_data.head()
raw_data.iloc[0,:], len(raw_data.iloc[0,:])
raw_data.iloc[0,0]

lemmizer = nltk.WordNetLemmatizer()



# note that i used the results from the word_punct tokenizer, but you can use any word tokenizer method



for w in doc_words_punct[0][31:55]:

        print(lemmizer.lemmatize(w), w)
# Lowercasing

for w in doc_words_punct[0][33:37]:

    print(w.lower(), w)
# CountVectorizer is a class; so `vectorizer` below represents an instance of that object.

c_vectorizer = CountVectorizer(ngram_range=(1,3), 

                             stop_words='english', 

                             max_df = 0.6, 

                             max_features=10000)



# call `fit` to build the vocabulary

c_vectorizer.fit(clean_abs)

# finally, call `transform` to convert text to a bag of words

c_data = c_vectorizer.transform(clean_abs)
# Topic Modeling

%env JOBLIB_TEMP_FOLDER=/tmp
def topic_mod_als(vectorizer, vect_data, topics=20, iters=5, no_top_words=50):

    

    """ use Latent Dirichlet Allocation to get topics"""



    mod = LatentDirichletAllocation(n_components=topics,

                                    max_iter=iters,

                                    random_state=42,

                                    learning_method='online',

                                    n_jobs=-1)

    

    mod_dat = mod.fit_transform(vect_data)

    

    

    # to display a list of topic words and their scores 

    #next step is to make this into a matrix with all 5k terms and their scores from the viz?  

    

    def display_topics(model, feature_names, no_top_words):

        for ix, topic in enumerate(model.components_):

            print("Topic ", ix)

            print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-no_top_words - 1:-1]]) + '\n')

    

    display_topics(mod, vectorizer.get_feature_names() , no_top_words)



    

    return mod, mod_dat

mod, mod_dat = topic_mod_als(c_vectorizer, 

                            c_data, 

                            topics=20, 

                            iters=10, 

                            no_top_words=15)  
# Visualization



import pyLDAvis, pyLDAvis.sklearn

from IPython.display import display 

    

# Setup to run in Jupyter notebooks

pyLDAvis.enable_notebook()



 # Create the visualization

vis = pyLDAvis.sklearn.prepare(mod, c_data, c_vectorizer,  sort_topics=False, mds='mmds')#, mds='tsne'



# Let's view it!

display(vis)
