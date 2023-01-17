import os

print(os.listdir("../input/noteevents1"))

# import library packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# input data from csv files or database sources

path1 = '../input/noteevents1/'

#pd.read_csv(path1+"ADMISSIONS.csv")

notes = pd.read_csv(path1+"NOTEEVENTS1.csv")

notes.head



# select text notes under category "Discharge summary"

notes_dis = notes.loc[notes.CATEGORY == 'Discharge summary']

# check if there are empty text notes

notes_dis.TEXT.isnull().sum() / len(notes_dis.TEXT) # no empty text notes



def preprocess_text(inputs):

# This function preprocesses the text by filling not a 

# number and replacing new lines ('\n') and carriage returns ('\r')

    inputs.TEXT = inputs.TEXT.fillna(' ')

    inputs.TEXT = inputs.TEXT.str.replace('\n', ' ')

    inputs.TEXT = inputs.TEXT.str.replace('\t', ' ')

    return inputs

# slice a part of data to see text preprocessing results

notes_dis_sample = preprocess_text(notes_dis.loc[:3,:])



# tokenize doctor's notes using nltk package and a token-optimizing function

from nltk import word_tokenize

word_tokenize(notes_dis_sample.TEXT[1])# initial tokens include punctuation and numbers

import string

def tokenizer_better(text):

    # tokenize the text by replacing punctuation and numbers with spaces and lowercase all words

    punc_list = string.punctuation+'0123456789'

    t = str.maketrans(dict.fromkeys(punc_list, " "))

    text = text.lower().translate(t)

    tokens = word_tokenize(text)

    return tokens

tokenizer_better(notes_dis_sample.TEXT[1])[40:60]

# using bag-of-words approach to count the numbers of each word

# 1. vectorize text using sci-learn package

from  sklearn.feature_extraction.text import CountVectorizer

model1 = CountVectorizer(tokenizer = tokenizer_better)

model1.fit(notes_dis_sample.TEXT)

sparse_vec = model1.transform(notes_dis_sample.TEXT) # sparse matrix, only non-zero values are kept

d = sparse_vec.toarray()

d.shape

# plot to see the most used words and select stop words

token_num_sum = np.squeeze(np.sum(sparse_vec,axis=0))

token_num_sum_df = pd.DataFrame(token_num_sum,columns=model1.get_feature_names()).sort_values(by=0,axis=1,ascending=False)

plot1 = pd.Series(token_num_sum_df.iloc[0],index=token_num_sum_df.columns)

ax = plot1[:50].plot(kind='bar', figsize=(10,6), width=.5, fontsize=14, rot=90,color = 'b')

ax.title.set_size(18)

plt.ylabel('count')

plt.show()

my_stop_words = ['the','and','to','of','was','with','a','on','in','for','name',

                 'is','patient','s','he','at','as','or','one','she','his','her','am',

                 'were','you','pt','pm','by','be','had','your','this','date',

                'from','there','an','that','p','are','have','has','h','but','o',

                'namepattern','which','every','also','d','q']

#include stop words in the vectorizing model

model2 = CountVectorizer(tokenizer = tokenizer_better,stop_words=my_stop_words)

model2.fit(notes_dis_sample.TEXT)

sparse_vec = model2.transform(notes_dis_sample.TEXT)

sparse_vec.toarray()

token_num_sum = np.squeeze(np.sum(sparse_vec,axis=0))

token_num_sum_df = pd.DataFrame(token_num_sum,columns=model2.get_feature_names()).sort_values(by=0,axis=1,ascending=False)

plot1 = pd.Series(token_num_sum_df.iloc[0],index=token_num_sum_df.columns)

ax = plot1[:50].plot(kind='bar', figsize=(10,6), width=.5, fontsize=14, rot=90,color = 'b')

ax.title.set_size(18)

plt.ylabel('count')

plt.show()

#path2 = R"""C:\Users\Administator\Desktop\NLP-key_extraction\NLP-doctors-notes\\"""

path2='../input/symptoms/'

with open(path2+'symptoms.txt','r') as f:

    symptom_words = f.readlines()

symptom_words_lib = [x.lstrip() for x in symptom_words[0].split(',')]

model3 = CountVectorizer(tokenizer = tokenizer_better,stop_words=my_stop_words)

model3.fit(symptom_words_lib)

sparse_vec = model3.transform(notes_dis_sample.TEXT)

sparse_vec.toarray()

token_num_sum = np.squeeze(np.sum(sparse_vec,axis=0))

token_num_sum_df = pd.DataFrame(token_num_sum,columns=model3.get_feature_names()).sort_values(by=0,axis=1,ascending=False)

plot1 = pd.Series(token_num_sum_df.iloc[0],index=token_num_sum_df.columns)

ax = plot1[:50].plot(kind='bar', figsize=(10,6), width=.5, fontsize=14, rot=90,color = 'b')

ax.title.set_size(18)

plt.ylabel('count')

plt.title('Symptom Words extraction')

plt.show()