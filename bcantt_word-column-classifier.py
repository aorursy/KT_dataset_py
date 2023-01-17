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

from nltk.tokenize import word_tokenize

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train_text
train_text = train.text.apply(word_tokenize)

test_text = test.text.apply(word_tokenize)
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



stop_words = set(stopwords.words( 'english' ))



filtered_train_text = []

filtered_test_text = []

for sentence in train_text:

    row_sentence = []

    for word in sentence:

        if word not in stop_words: 

            row_sentence.append(word.lower()) 

            

    filtered_train_text.append(row_sentence)

    

for sentence in test_text:

    row_sentence = []

    for word in sentence:

        if word not in stop_words: 

            row_sentence.append(word.lower()) 

            

    filtered_test_text.append(row_sentence)
last_train = []

last_test = []

for sentence in filtered_train_text:

    row_sentence = []

    for word in sentence:

        if word.isalpha(): 

            row_sentence.append(word.lower()) 

            

    last_train.append(row_sentence)

    

for sentence in filtered_test_text:

    row_sentence = []

    for word in sentence:

        if word.isalpha(): 

            row_sentence.append(word.lower()) 

            

    last_test.append(row_sentence)
last_train = []

last_test = []

for sentence in filtered_train_text:

    row_sentence = []

    for word in sentence:

        if word.isalpha(): 

            row_sentence.append(word.lower()) 

            

    last_train.append(row_sentence)

    

for sentence in filtered_test_text:

    row_sentence = []

    for word in sentence:

        if word.isalpha(): 

            row_sentence.append(word.lower()) 

            

    last_test.append(row_sentence)
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
for sentence in last_train:

    for i in range(len(sentence)):

        sentence[i] = porter.stem(sentence[i])

        

for sentence in last_test:

    for i in range(len(sentence)):

        sentence[i] = porter.stem(sentence[i])
headers_train = ['col' + str(i) for i in range(27)] 
train_df = pd.DataFrame(last_train, columns=headers_train)

test_df =  pd.DataFrame(last_test, columns=headers_train)
train_df = train_df.fillna('None')

test_df = test_df.fillna('None')
all_data = pd.concat([train_df,test_df])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for name in all_data.columns:

    all_data[name] = all_data[name].astype(str)

    le.fit(all_data.values.flatten())

    train_df[name] = le.transform(train_df[name])

    test_df[name] = le.transform(test_df[name])

    print(name)
train_df['target'] = train['target']
train_df
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

    disaster_tweets_sum[common_key] = disaster_tweets_sum[common_key] + non_disaster_tweets_sum[common_key] 
for common_key in set(disaster_tweets_copy) & set(non_disaster_tweets_copy):

    non_disaster_tweets_copy[common_key] = non_disaster_tweets_copy[common_key]/disaster_tweets_copy[common_key]
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

train_df['positive_score'] = positive_score_train

train_df['negative_score'] = negative_score_train

train_df['number_of_occurance_positive'] = number_of_occurance_positive_train

train_df['number_of_occurance_negative'] = number_of_occurance_negative_train

train_df['positive_ratios'] = positive_ratios_train

train_df['negative_ratios'] = negative_ratios_train

train_df['positive_ratios_mult'] = positive_ratios_mult_train

train_df['negative_ratios_mult'] = negative_ratios_mult_train

train_df['sum_voc_positive'] = sum_voc_positive_train

train_df['sum_voc_negative']= sum_voc_negative_train



train_df['non_positive_score'] = non_positive_score_train

train_df['non_negative_score'] = non_negative_score_train

train_df['non_number_of_occurance_positive'] = non_number_of_occurance_positive_train

train_df['non_number_of_occurance_negative'] = non_number_of_occurance_negative_train

train_df['non_positive_ratios'] = non_positive_ratios_train

train_df['non_negative_ratios'] = non_negative_ratios_train

train_df['non_positive_ratios_mult'] = non_positive_ratios_mult_train

train_df['non_negative_ratios_mult'] = non_negative_ratios_mult_train



train_df['diff_positive_score'] = diff_positive_score_train

train_df['diff_negative_score'] = diff_negative_score_train

train_df['diff_number_of_occurance_positive'] = diff_number_of_occurance_positive_train

train_df['diff_number_of_occurance_negative'] = diff_number_of_occurance_negative_train

train_df['diff_positive_ratios'] = diff_positive_ratios_train

train_df['diff_negative_ratios'] = diff_negative_ratios_train

train_df['diff_positive_ratios_mult'] = diff_positive_ratios_mult_train

train_df['diff_negative_ratios_mult'] = diff_negative_ratios_mult_train

train_df['diff_sum_voc_positive'] = diff_sum_voc_positive_train

train_df['diff_sum_voc_negative'] = diff_sum_voc_negative_train



train_df['diff_non_positive_score'] = diff_non_positive_score_train

train_df['diff_non_negative_score'] = diff_non_negative_score_train

train_df['diff_non_number_of_occurance_positive'] = diff_non_number_of_occurance_positive_train

train_df['diff_non_number_of_occurance_negative'] = diff_non_number_of_occurance_negative_train

train_df['diff_non_positive_ratios'] = diff_non_positive_ratios_train

train_df['diff_non_negative_ratios'] = diff_non_negative_ratios_train

train_df['diff_non_positive_ratios_mult'] = diff_non_positive_ratios_mult_train

train_df['diff_non_negative_ratios_mult'] = diff_non_negative_ratios_mult_train





test_df['positive_score'] = positive_score_test

test_df['negative_score'] = negative_score_test

test_df['number_of_occurance_positive'] = number_of_occurance_positive_test

test_df['number_of_occurance_negative'] = number_of_occurance_negative_test

test_df['positive_ratios'] = positive_ratios_test

test_df['negative_ratios'] = negative_ratios_test

test_df['positive_ratios_mult'] = positive_ratios_mult_test

test_df['negative_ratios_mult'] = negative_ratios_mult_test

test_df['sum_voc_positive'] = sum_voc_positive_test 

test_df['sum_voc_negative']= sum_voc_negative_test 





test_df['non_positive_score'] = non_positive_score_test

test_df['non_negative_score'] = non_negative_score_test

test_df['non_number_of_occurance_positive'] = non_number_of_occurance_positive_test

test_df['non_number_of_occurance_negative'] = non_number_of_occurance_negative_test

test_df['non_positive_ratios'] = non_positive_ratios_test

test_df['non_negative_ratios'] = non_negative_ratios_test

test_df['non_positive_ratios_mult'] = non_positive_ratios_mult_test

test_df['non_negative_ratios_mult'] = non_negative_ratios_mult_test



test_df['diff_positive_score'] = diff_positive_score_test

test_df['diff_negative_score'] = diff_negative_score_test

test_df['diff_number_of_occurance_positive'] = diff_number_of_occurance_positive_test

test_df['diff_number_of_occurance_negative'] = diff_number_of_occurance_negative_test

test_df['diff_positive_ratios'] = diff_positive_ratios_test

test_df['diff_negative_ratios'] = diff_negative_ratios_test

test_df['diff_positive_ratios_mult'] = diff_positive_ratios_mult_test

test_df['diff_negative_ratios_mult'] = diff_negative_ratios_mult_test

test_df['diff_sum_voc_positive'] = diff_sum_voc_positive_test

test_df['diff_sum_voc_negative'] = diff_sum_voc_negative_test



test_df['diff_non_positive_score'] = diff_non_positive_score_test

test_df['diff_non_negative_score'] = diff_non_negative_score_test

test_df['diff_non_number_of_occurance_positive'] = diff_non_number_of_occurance_positive_test

test_df['diff_non_number_of_occurance_negative'] = diff_non_number_of_occurance_negative_test

test_df['diff_non_positive_ratios'] = diff_non_positive_ratios_test

test_df['diff_non_negative_ratios'] = diff_non_negative_ratios_test

test_df['diff_non_positive_ratios_mult'] = diff_non_positive_ratios_mult_test

test_df['diff_non_negative_ratios_mult'] = diff_non_negative_ratios_mult_test
train_df.replace([np.inf, -np.inf], np.nan).dropna(subset = train_df.columns, how="all")

test_df.replace([np.inf, -np.inf], np.nan).dropna(subset = test_df.columns, how="all")
X = train_df.drop('target',axis = 1)

y = train_df['target']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)

X_normalized = scaler.transform(X)



test_X = test_df

scaler.fit(test_X)

test_X_normalized = scaler.transform(test_X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.20, random_state=42)
classifier = RandomForestClassifier()

classifier.fit(X_normalized, y)

predictions = classifier.predict(test_X_normalized)
sample_submission['target'] = predictions
sample_submission
test['target'] = predictions
test[['text','target']].head(60)
sample_submission.to_csv('submission.csv',index=False)