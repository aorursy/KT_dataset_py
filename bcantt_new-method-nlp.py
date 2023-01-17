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
import pandas as pd

import os

import re

import spacy

from sklearn import preprocessing

from gensim.models.phrases import Phrases, Phraser

from time import time 

import multiprocessing

from gensim.models import Word2Vec

import bokeh.plotting as bp

from bokeh.models import HoverTool, BoxSelectTool

from bokeh.plotting import figure, show, output_notebook

from sklearn.manifold import TSNE

import numpy as np

from sklearn.preprocessing import scale

import keras 

from keras.models import Sequential, Model 

from keras import layers

from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding

from keras.layers.merge import Concatenate

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud

from nltk.tokenize import RegexpTokenizer

from sklearn.metrics import confusion_matrix

from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt

import gensim

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import sent_tokenize, word_tokenize 

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
disaster_tweets = train.loc[train['target'] == 1]['text'] 

non_disaster_tweets = train.loc[train['target'] == 0]['text'] 
vectorizer = CountVectorizer(min_df=0, lowercase=True)

vectorizer.fit(disaster_tweets)

disaster_voc = vectorizer.vocabulary_

disaster_tweets_copy = disaster_voc.copy()

disaster_tweets_sum = disaster_voc.copy()

vectorizer.fit(non_disaster_tweets)

non_disaster_voc = vectorizer.vocabulary_

non_disaster_tweets_copy = non_disaster_voc.copy()

non_disaster_tweets_sum = non_disaster_voc.copy()

diff_disaster_tweets = set(disaster_voc) -  set(non_disaster_voc)

diff_non_disaster_tweets = set(non_disaster_voc) -  set(disaster_voc)
for common_key in set(disaster_voc) & set(non_disaster_voc):

    disaster_voc[common_key] = disaster_voc[common_key] - non_disaster_voc[common_key] 
for common_key in set(disaster_voc) & set(non_disaster_voc):

    non_disaster_voc[common_key] = non_disaster_voc[common_key] - disaster_voc[common_key] 
for common_key in set(disaster_tweets_copy) & set(non_disaster_tweets_copy):

    disaster_tweets_copy[common_key] = disaster_tweets_copy[common_key]/non_disaster_tweets_copy[common_key] 
for common_key in set(disaster_tweets_copy) & set(non_disaster_tweets_copy):

    non_disaster_tweets_copy[common_key] = non_disaster_tweets_copy[common_key]/disaster_tweets_copy[common_key]
for common_key in set(disaster_tweets_copy) & set(non_disaster_tweets_copy):

    disaster_tweets_sum[common_key] = disaster_tweets_sum[common_key] + non_disaster_tweets_sum[common_key] 
positive_score_train = []

negative_score_train = []

number_of_occurance_positive_train= []

number_of_occurance_negative_train = []

positive_ratios_train = []

negative_ratios_train = []

positive_ratios_mult_train = []

negative_ratios_mult_train = []

sum_voc_positive_train = []

sum_voc_negative_train = []



non_positive_score_train = []

non_negative_score_train = []

non_number_of_occurance_positive_train= []

non_number_of_occurance_negative_train = []

non_positive_ratios_train = []

non_negative_ratios_train = []

non_positive_ratios_mult_train = []

non_negative_ratios_mult_train = []



diff_positive_score_train = []

diff_negative_score_train = []

diff_number_of_occurance_positive_train = []

diff_number_of_occurance_negative_train = []

diff_positive_ratios_train = []

diff_negative_ratios_train = []

diff_positive_ratios_mult_train = []

diff_negative_ratios_mult_train = []

diff_sum_voc_positive_train = []

diff_sum_voc_negative_train = []



diff_non_positive_score_train = []

diff_non_negative_score_train = []

diff_non_number_of_occurance_positive_train= []

diff_non_number_of_occurance_negative_train = []

diff_non_positive_ratios_train = []

diff_non_negative_ratios_train = []

diff_non_positive_ratios_mult_train = []

diff_non_negative_ratios_mult_train = []



for i in range(len(train['text'].values)):

    score_positive = 0

    score_negative = 0

    number_positive = 0

    number_negative = 0

    positive_ratios = 0

    negative_ratios = 0

    positive_ratios_mult = 1

    negative_ratios_mult = 1

    sum_voc_positive = 0

    sum_voc_negative = 0

    

    non_score_positive = 0

    non_score_negative = 0

    non_number_positive = 0

    non_number_negative = 0

    non_positive_ratios = 0

    non_negative_ratios = 0

    non_positive_ratios_mult = 1

    non_negative_ratios_mult = 1

    

    diff_score_positive = 0

    diff_score_negative = 0

    diff_number_positive = 0

    diff_number_negative = 0

    diff_positive_ratios = 0

    diff_negative_ratios = 0

    diff_positive_ratios_mult = 1

    diff_negative_ratios_mult = 1

    diff_sum_voc_positive = 0

    diff_sum_voc_negative = 0

    

    diff_non_score_positive = 0

    diff_non_score_negative = 0

    diff_non_number_positive = 0

    diff_non_number_negative = 0

    diff_non_positive_ratios = 0

    diff_non_negative_ratios = 0

    diff_non_positive_ratios_mult = 1

    diff_non_negative_ratios_mult = 1

   

    for common_key in set(disaster_voc) & set([x.lower() for x in word_tokenize(train['text'].values[i])]):

        if disaster_voc[common_key] >= 0:

            score_positive += disaster_voc[common_key]

            number_positive += 1

            positive_ratios += disaster_tweets_copy[common_key]

            positive_ratios_mult *= disaster_tweets_copy[common_key]

            sum_voc_positive += disaster_tweets_sum[common_key]

           

        else:

            score_negative += disaster_voc[common_key]

            number_negative += 1

            negative_ratios += disaster_tweets_copy[common_key]

            negative_ratios_mult *= disaster_tweets_copy[common_key]

            sum_voc_negative += disaster_tweets_sum[common_key]

            

            

            

    for common_key in set(non_disaster_voc) & set([x.lower() for x in word_tokenize(train['text'].values[i])]):

        if non_disaster_voc[common_key] >= 0:

            non_score_positive += non_disaster_voc[common_key]

            non_number_positive += 1

            non_positive_ratios += non_disaster_tweets_copy[common_key]

            non_positive_ratios_mult *= non_disaster_tweets_copy[common_key]

        else:

            non_score_negative += non_disaster_voc[common_key]

            non_number_negative += 1

            non_negative_ratios += non_disaster_tweets_copy[common_key]

            non_negative_ratios_mult *= non_disaster_tweets_copy[common_key]

            

    for common_key in set(diff_disaster_tweets) &  set([x.lower() for x in word_tokenize(train['text'].values[i])]):

        if disaster_voc[common_key] >= 0:

            diff_score_positive += disaster_voc[common_key]

            diff_number_positive += 1

            diff_positive_ratios += disaster_tweets_copy[common_key]

            diff_positive_ratios_mult *= disaster_tweets_copy[common_key]

            diff_sum_voc_positive += disaster_tweets_sum[common_key]

              

        else:

            diff_score_negative += disaster_voc[common_key]

            diff_number_negative += 1

            diff_negative_ratios += disaster_tweets_copy[common_key]

            diff_negative_ratios_mult *= disaster_tweets_copy[common_key]

            diff_sum_voc_negative += disaster_tweets_sum[common_key]

            

    for common_key in set(diff_non_disaster_tweets)  & set([x.lower() for x in word_tokenize(train['text'].values[i])]):

        if non_disaster_voc[common_key] >= 0:

            diff_non_score_positive += non_disaster_voc[common_key]

            diff_non_number_positive += 1

            diff_non_positive_ratios += non_disaster_tweets_copy[common_key]

            diff_non_positive_ratios_mult *= non_disaster_tweets_copy[common_key]

        else:

            diff_non_score_negative += non_disaster_voc[common_key]

            diff_non_number_negative += 1

            diff_non_negative_ratios += non_disaster_tweets_copy[common_key]

            diff_non_negative_ratios_mult *= non_disaster_tweets_copy[common_key]



        

    positive_score_train.append(score_positive)

    negative_score_train.append(score_negative)

    number_of_occurance_positive_train.append(number_positive)

    number_of_occurance_negative_train.append(number_negative)

    positive_ratios_train.append(positive_ratios)

    negative_ratios_train.append(negative_ratios)

    positive_ratios_mult_train.append(positive_ratios_mult)

    negative_ratios_mult_train.append(negative_ratios_mult)

    sum_voc_positive_train.append(sum_voc_positive)

    sum_voc_negative_train.append(sum_voc_negative)

    

    non_positive_score_train.append(non_score_positive)

    non_negative_score_train.append(non_score_negative)

    non_number_of_occurance_positive_train.append(non_number_positive)

    non_number_of_occurance_negative_train.append(non_number_negative)

    non_positive_ratios_train.append(non_positive_ratios)

    non_negative_ratios_train.append(non_negative_ratios)

    non_positive_ratios_mult_train.append(non_positive_ratios_mult)

    non_negative_ratios_mult_train.append(non_negative_ratios_mult)

    

    diff_positive_score_train.append(diff_score_positive)

    diff_negative_score_train.append(diff_score_negative)

    diff_number_of_occurance_positive_train.append(diff_number_positive)

    diff_number_of_occurance_negative_train.append(diff_number_negative)

    diff_positive_ratios_train.append(diff_positive_ratios)

    diff_negative_ratios_train.append(diff_negative_ratios)

    diff_positive_ratios_mult_train.append(diff_positive_ratios_mult)

    diff_negative_ratios_mult_train.append(diff_negative_ratios_mult)

    diff_sum_voc_positive_train.append(diff_sum_voc_positive)

    diff_sum_voc_negative_train.append(diff_sum_voc_negative)

    

    diff_non_positive_score_train.append(diff_non_score_positive)

    diff_non_negative_score_train.append(diff_non_score_negative)

    diff_non_number_of_occurance_positive_train.append(diff_non_number_positive)

    diff_non_number_of_occurance_negative_train.append(diff_non_number_negative)

    diff_non_positive_ratios_train.append(non_positive_ratios)

    diff_non_negative_ratios_train.append(diff_non_negative_ratios)

    diff_non_positive_ratios_mult_train.append(diff_non_positive_ratios_mult)

    diff_non_negative_ratios_mult_train.append(diff_non_negative_ratios_mult)

    

    

                

                

        
positive_score_test = []

negative_score_test = []

number_of_occurance_positive_test = []

number_of_occurance_negative_test = []

positive_ratios_test = []

negative_ratios_test = []

positive_ratios_mult_test = []

negative_ratios_mult_test = []

sum_voc_positive_test = []

sum_voc_negative_test = []



non_positive_score_test = []

non_negative_score_test = []

non_number_of_occurance_positive_test= []

non_number_of_occurance_negative_test = []

non_positive_ratios_test = []

non_negative_ratios_test = []

non_positive_ratios_mult_test = []

non_negative_ratios_mult_test = []



diff_positive_score_test = []

diff_negative_score_test = []

diff_number_of_occurance_positive_test = []

diff_number_of_occurance_negative_test = []

diff_positive_ratios_test = []

diff_negative_ratios_test = []

diff_positive_ratios_mult_test = []

diff_negative_ratios_mult_test = []

diff_sum_voc_positive_test = []

diff_sum_voc_negative_test = []



diff_non_positive_score_test = []

diff_non_negative_score_test = []

diff_non_number_of_occurance_positive_test= []

diff_non_number_of_occurance_negative_test = []

diff_non_positive_ratios_test = []

diff_non_negative_ratios_test = []

diff_non_positive_ratios_mult_test = []

diff_non_negative_ratios_mult_test = []



for i in range(len(test['text'].values)):

    score_positive = 0

    score_negative = 0

    number_positive = 0

    number_negative = 0

    positive_ratios = 0

    negative_ratios = 0

    positive_ratios_mult = 1

    negative_ratios_mult = 1

    sum_voc_positive = 0

    sum_voc_negative = 0

    

    non_score_positive = 0

    non_score_negative = 0

    non_number_positive = 0

    non_number_negative = 0

    non_positive_ratios = 0

    non_negative_ratios = 0

    non_positive_ratios_mult = 1

    non_negative_ratios_mult = 1

    

    diff_score_positive = 0

    diff_score_negative = 0

    diff_number_positive = 0

    diff_number_negative = 0

    diff_positive_ratios = 0

    diff_negative_ratios = 0

    diff_positive_ratios_mult = 1

    diff_negative_ratios_mult = 1

    diff_sum_voc_positive = 0

    diff_sum_voc_negative = 0

    

    diff_non_score_positive = 0

    diff_non_score_negative = 0

    diff_non_number_positive = 0

    diff_non_number_negative = 0

    diff_non_positive_ratios = 0

    diff_non_negative_ratios = 0

    diff_non_positive_ratios_mult = 1

    diff_non_negative_ratios_mult = 1

    

    for common_key in set(disaster_voc) & set([x.lower() for x in word_tokenize(test['text'].values[i])]):

        if disaster_voc[common_key] >= 0:

            score_positive += disaster_voc[common_key]

            number_positive += 1

            positive_ratios += disaster_tweets_copy[common_key]

            positive_ratios_mult *= disaster_tweets_copy[common_key]

            sum_voc_positive += disaster_tweets_sum[common_key]

              

        else:

            score_negative += disaster_voc[common_key]

            number_negative += 1

            negative_ratios += disaster_tweets_copy[common_key]

            negative_ratios_mult *= disaster_tweets_copy[common_key]

            sum_voc_negative += disaster_tweets_sum[common_key]



            

    for common_key in set(non_disaster_voc) & set([x.lower() for x in word_tokenize(test['text'].values[i])]):

        if non_disaster_voc[common_key] >= 0:

            non_score_positive += non_disaster_voc[common_key]

            non_number_positive += 1

            non_positive_ratios += non_disaster_tweets_copy[common_key]

            non_positive_ratios_mult *= non_disaster_tweets_copy[common_key]

        else:

            non_score_negative += non_disaster_voc[common_key]

            non_number_negative += 1

            non_negative_ratios += non_disaster_tweets_copy[common_key]

            non_negative_ratios_mult *= non_disaster_tweets_copy[common_key]

            

            

    for common_key in set(diff_disaster_tweets) &  set([x.lower() for x in word_tokenize(test['text'].values[i])]):

        if disaster_voc[common_key] >= 0:

            diff_score_positive += disaster_voc[common_key]

            diff_number_positive += 1

            diff_positive_ratios += disaster_tweets_copy[common_key]

            diff_positive_ratios_mult *= disaster_tweets_copy[common_key]

            diff_sum_voc_positive += disaster_tweets_sum[common_key]

              

        else:

            diff_score_negative += disaster_voc[common_key]

            diff_number_negative += 1

            diff_negative_ratios += disaster_tweets_copy[common_key]

            diff_negative_ratios_mult *= disaster_tweets_copy[common_key]

            diff_sum_voc_negative += disaster_tweets_sum[common_key]

            

    for common_key in set(diff_non_disaster_tweets)  & set([x.lower() for x in word_tokenize(test['text'].values[i])]):

        if non_disaster_voc[common_key] >= 0:

            diff_non_score_positive += non_disaster_voc[common_key]

            diff_non_number_positive += 1

            diff_non_positive_ratios += non_disaster_tweets_copy[common_key]

            diff_non_positive_ratios_mult *= non_disaster_tweets_copy[common_key]

        else:

            diff_non_score_negative += non_disaster_voc[common_key]

            diff_non_number_negative += 1

            diff_non_negative_ratios += non_disaster_tweets_copy[common_key]

            diff_non_negative_ratios_mult *= non_disaster_tweets_copy[common_key]

        

    positive_score_test.append(score_positive)

    negative_score_test.append(score_negative)

    number_of_occurance_positive_test.append(number_positive)

    number_of_occurance_negative_test.append(number_negative)

    positive_ratios_test.append(positive_ratios)

    negative_ratios_test.append(negative_ratios)

    positive_ratios_mult_test.append(positive_ratios_mult)

    negative_ratios_mult_test.append(negative_ratios_mult)

    sum_voc_positive_test.append(sum_voc_positive)

    sum_voc_negative_test.append(sum_voc_negative)

    

    non_positive_score_test.append(non_score_positive)

    non_negative_score_test.append(non_score_negative)

    non_number_of_occurance_positive_test.append(non_number_positive)

    non_number_of_occurance_negative_test.append(non_number_negative)

    non_positive_ratios_test.append(non_positive_ratios)

    non_negative_ratios_test.append(non_negative_ratios)

    non_positive_ratios_mult_test.append(non_positive_ratios_mult)

    non_negative_ratios_mult_test.append(non_negative_ratios_mult)

    

    diff_positive_score_test.append(diff_score_positive)

    diff_negative_score_test.append(diff_score_negative)

    diff_number_of_occurance_positive_test.append(diff_number_positive)

    diff_number_of_occurance_negative_test.append(diff_number_negative)

    diff_positive_ratios_test.append(diff_positive_ratios)

    diff_negative_ratios_test.append(diff_negative_ratios)

    diff_positive_ratios_mult_test.append(diff_positive_ratios_mult)

    diff_negative_ratios_mult_test.append(diff_negative_ratios_mult)

    diff_sum_voc_positive_test.append(diff_sum_voc_positive)

    diff_sum_voc_negative_test.append(diff_sum_voc_negative)

    

    diff_non_positive_score_test.append(diff_non_score_positive)

    diff_non_negative_score_test.append(diff_non_score_negative)

    diff_non_number_of_occurance_positive_test.append(diff_non_number_positive)

    diff_non_number_of_occurance_negative_test.append(diff_non_number_negative)

    diff_non_positive_ratios_test.append(non_positive_ratios)

    diff_non_negative_ratios_test.append(diff_non_negative_ratios)

    diff_non_positive_ratios_mult_test.append(diff_non_positive_ratios_mult)

    diff_non_negative_ratios_mult_test.append(diff_non_negative_ratios_mult)



                
train['positive_score'] = positive_score_train

train['negative_score'] = negative_score_train

train['number_of_occurance_positive'] = number_of_occurance_positive_train

train['number_of_occurance_negative'] = number_of_occurance_negative_train

train['positive_ratios'] = positive_ratios_train

train['negative_ratios'] = negative_ratios_train

train['positive_ratios_mult'] = positive_ratios_mult_train

train['negative_ratios_mult'] = negative_ratios_mult_train

train['sum_voc_positive'] = sum_voc_positive_train

train['sum_voc_negative']= sum_voc_negative_train



train['non_positive_score'] = non_positive_score_train

train['non_negative_score'] = non_negative_score_train

train['non_number_of_occurance_positive'] = non_number_of_occurance_positive_train

train['non_number_of_occurance_negative'] = non_number_of_occurance_negative_train

train['non_positive_ratios'] = non_positive_ratios_train

train['non_negative_ratios'] = non_negative_ratios_train

train['non_positive_ratios_mult'] = non_positive_ratios_mult_train

train['non_negative_ratios_mult'] = non_negative_ratios_mult_train



train['diff_positive_score'] = diff_positive_score_train

train['diff_negative_score'] = diff_negative_score_train

train['diff_number_of_occurance_positive'] = diff_number_of_occurance_positive_train

train['diff_number_of_occurance_negative'] = diff_number_of_occurance_negative_train

train['diff_positive_ratios'] = diff_positive_ratios_train

train['diff_negative_ratios'] = diff_negative_ratios_train

train['diff_positive_ratios_mult'] = diff_positive_ratios_mult_train

train['diff_negative_ratios_mult'] = diff_negative_ratios_mult_train

train['diff_sum_voc_positive'] = diff_sum_voc_positive_train

train['diff_sum_voc_negative'] = diff_sum_voc_negative_train



train['diff_non_positive_score'] = diff_non_positive_score_train

train['diff_non_negative_score'] = diff_non_negative_score_train

train['diff_non_number_of_occurance_positive'] = diff_non_number_of_occurance_positive_train

train['diff_non_number_of_occurance_negative'] = diff_non_number_of_occurance_negative_train

train['diff_non_positive_ratios'] = diff_non_positive_ratios_train

train['diff_non_negative_ratios'] = diff_non_negative_ratios_train

train['diff_non_positive_ratios_mult'] = diff_non_positive_ratios_mult_train

train['diff_non_negative_ratios_mult'] = diff_non_negative_ratios_mult_train





test['positive_score'] = positive_score_test

test['negative_score'] = negative_score_test

test['number_of_occurance_positive'] = number_of_occurance_positive_test

test['number_of_occurance_negative'] = number_of_occurance_negative_test

test['positive_ratios'] = positive_ratios_test

test['negative_ratios'] = negative_ratios_test

test['positive_ratios_mult'] = positive_ratios_mult_test

test['negative_ratios_mult'] = negative_ratios_mult_test

test['sum_voc_positive'] = sum_voc_positive_test 

test['sum_voc_negative']= sum_voc_negative_test 





test['non_positive_score'] = non_positive_score_test

test['non_negative_score'] = non_negative_score_test

test['non_number_of_occurance_positive'] = non_number_of_occurance_positive_test

test['non_number_of_occurance_negative'] = non_number_of_occurance_negative_test

test['non_positive_ratios'] = non_positive_ratios_test

test['non_negative_ratios'] = non_negative_ratios_test

test['non_positive_ratios_mult'] = non_positive_ratios_mult_test

test['non_negative_ratios_mult'] = non_negative_ratios_mult_test



test['diff_positive_score'] = diff_positive_score_test

test['diff_negative_score'] = diff_negative_score_test

test['diff_number_of_occurance_positive'] = diff_number_of_occurance_positive_test

test['diff_number_of_occurance_negative'] = diff_number_of_occurance_negative_test

test['diff_positive_ratios'] = diff_positive_ratios_test

test['diff_negative_ratios'] = diff_negative_ratios_test

test['diff_positive_ratios_mult'] = diff_positive_ratios_mult_test

test['diff_negative_ratios_mult'] = diff_negative_ratios_mult_test

test['diff_sum_voc_positive'] = diff_sum_voc_positive_test

test['diff_sum_voc_negative'] = diff_sum_voc_negative_test



test['diff_non_positive_score'] = diff_non_positive_score_test

test['diff_non_negative_score'] = diff_non_negative_score_test

test['diff_non_number_of_occurance_positive'] = diff_non_number_of_occurance_positive_test

test['diff_non_number_of_occurance_negative'] = diff_non_number_of_occurance_negative_test

test['diff_non_positive_ratios'] = diff_non_positive_ratios_test

test['diff_non_negative_ratios'] = diff_non_negative_ratios_test

test['diff_non_positive_ratios_mult'] = diff_non_positive_ratios_mult_test

test['diff_non_negative_ratios_mult'] = diff_non_negative_ratios_mult_test
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

vectorizer.fit(train['text'])

feature_text_train = vectorizer.transform(train['text'])

feature_text_test = vectorizer.transform(test['text'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidfconverter = TfidfTransformer()

tfidfconverter.fit(feature_text_train)

feature_text_train = tfidfconverter.transform(feature_text_train).toarray()



tfidfconverter.fit(feature_text_test)

feature_text_test = tfidfconverter.transform(feature_text_test).toarray()

feature_text_train = pd.DataFrame(feature_text_train,columns = ['col' + str(i ) for i in range(1500)])

feature_text_test = pd.DataFrame(feature_text_test,columns = ['col' + str(i ) for i in range(1500)])
train = pd.concat([train.reset_index(drop=True),feature_text_train.reset_index(drop=True)], axis=1)

test = pd.concat([test.reset_index(drop=True),feature_text_test.reset_index(drop=True)], axis=1)
train.replace([np.inf, -np.inf], np.nan).dropna(subset = train.columns, how="all")

test.replace([np.inf, -np.inf], np.nan).dropna(subset = test.columns, how="all")
train.groupby('target').mean()
X = train.drop(['id','keyword','location','target','text'],axis = 1)

y = train['target']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)

X_normalized = scaler.transform(X)



test_X = test.drop(['text','id','keyword','location'],axis = 1)

scaler.fit(test_X)

test_X_normalized = scaler.transform(test_X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.33, random_state=42)




classifier = RandomForestClassifier()

classifier.fit(X_normalized, y)

predictions = classifier.predict(test_X_normalized)

sample_submission['target'] = predictions
test['target'] = predictions
test[['text','target']].head(60)
sample_submission.head(60)
sample_submission.to_csv('submission.csv',index=False)