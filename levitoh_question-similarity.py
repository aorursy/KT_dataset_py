# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#

# read sets into dataframes

df_train = pd.read_csv('../input/train.csv')

# for running local

#df_train = pd.read_csv('data/train.csv')

df_test = pd.read_csv('../input/test.csv')

#

# basic information about the dataset

print("there are {0} question pairs in training and {1} columns".format(df_train.shape[0], df_train.shape[1]))

#print("there are {0} question pairs in testing and {1} columns".format(df_test.shape[0], df_test.shape[1]))

print("-------- TRAIN DATA TYPES --------")

print(df_train.dtypes)

#print("-------- TEST DATA TYPES --------")

#print(df_test.dtypes)

print("-------- TRAIN MISSING VALUES --------")

print(df_train.isnull().sum())

#print("-------- TEST MISSING VALUES --------")

#print(df_test.isnull().sum())
#

# there are a small number of missing values, first i will see if it's just error and replace

# based on the question id

missingq1_qid1 = df_train.loc[df_train['question1'].isnull()]['qid1'].tolist()[0]

missingq2_qid2 = df_train.loc[df_train['question2'].isnull()]['qid2'].tolist()[0]   # this is the same question so just one item 

print("id for missing q1: {0}, id for missing q2: {1}".format(missingq1_qid1, missingq2_qid2))

#

# retrieve from dataset questions with previous ids (question 1 first)

if len(df_train.loc[(df_train['qid1'] == missingq1_qid1) & (df_train['id'] != df_train.loc[df_train['question1'].isnull()]['id'].tolist()[0])]) == 0:

    print("no question to replace missing q1")

else:

    print("replace missing q1 with question id: {}".format(df_train.loc[(df_train['qid1'] == missingq1_qid1) & (df_train['id'] != df_train.loc[df_train['question1'].isnull()]['id'].tolist()[0])]['id']))

#

# drop the missing values

df_train.dropna(inplace=True)

df_train = df_train.reset_index(drop=True)

print("-------- TRAIN MISSING VALUES --------")

print(df_train.isnull().sum())

print(df_train.shape)
#

# as is_duplicate is our target and as a binary column we can compute the mean to get

# the percentage of duplicate question in the dataset

print("percentage of duplicate questions in dataset: {0}%".format(df_train['is_duplicate'].mean()*100))

print("-------- COUNT, UNIQUE, TOP AND FREQUENCY FOR QUESTIONS 1 --------")

print(df_train.qid1.astype(str).describe())

print("-------- COUNT, UNIQUE, TOP AND FREQUENCY FOR QUESTIONS 2 --------")

print(df_train.qid2.astype(str).describe())

print("top question 1 in dataset: \'{0}\'".format(df_train.loc[df_train['qid1'] == 8461, 'question1'].iloc[0]))

print("top question 2 in dataset: \'{0}\'".format(df_train.loc[df_train['qid2'] == 30782, 'question2'].iloc[0]))

print("-------- PERCENTAGE OF UNIQUE QUESTIONS --------")

print("Q1: {}%".format(df_train.qid1.astype(str).describe()['unique']/df_train.qid1.astype(str).describe()['count']*100))

print("Q2: {}%".format(df_train.qid2.astype(str).describe()['unique']/df_train.qid2.astype(str).describe()['count']*100))
#

# visualization on unique questions and duplicate questions

# how many repeated questions q1 are duplicated

# how many repeated questions q2 are duplicated

# how many repeated questions q1 and q2 are duplicated

print(df_train.loc[df_train.duplicated('qid1')].is_duplicate.describe())

print(df_train.loc[df_train.duplicated('qid2')].is_duplicate.describe())

print(df_train.loc[df_train.duplicated('qid1') & df_train.duplicated('qid2')].is_duplicate.describe())
#

# visualization on unique questions and duplicate questions

# how many unique questions q1 are duplicated

# how many unique questions q2 are duplicated

# how many unique questions q1 and q2 are duplicated

print(df_train.loc[df_train.duplicated('qid1') == False].is_duplicate.describe())

print(df_train.loc[df_train.duplicated('qid2') == False].is_duplicate.describe())

print(df_train.loc[(df_train.duplicated('qid1') == False) & (df_train.duplicated('qid2') == False)].is_duplicate.describe())
#

# number of words per question (withouth pre-processing)

import spacy

nlp = spacy.load('en')

#

# tokenize words in question1 and quetion2 and get the length of tokens 

df_train['length_question1'] = df_train['question1'].apply(lambda x: len(nlp(x,  disable=['parser', 'tagger', 'ner'])))

df_train['length_question2'] = df_train['question2'].apply(lambda x: len(nlp(x,  disable=['parser', 'tagger', 'ner'])))
#

# Visualizing the distribution of variables in dataset

%matplotlib inline



import numpy as np

import pandas as pd

from scipy import stats, integrate

import matplotlib.pyplot as plt



import seaborn as sns

sns.set(color_codes=True)



np.random.seed(sum(map(ord, "distributions")))



fig, ax = plt.subplots(1,2,figsize=(30,15))

# frequency density on y axis

sns.distplot(ax=ax[0], bins=50, a=df_train['length_question1'])

sns.distplot(ax=ax[1], bins=50, a=df_train['length_question2'])

print("question 1 length mean: {0} and median: {1}".format(df_train['length_question1'].mean(), df_train['length_question1'].median()))

print("question 2 length mean: {0} and median: {1}".format(df_train['length_question2'].mean(), df_train['length_question2'].median()))

print("maximum length for question 1: {0}, question: \n{2}\n\nmaximum length for question 2: {1}, question: \n{3}".format(df_train['length_question1'].max(), df_train['length_question2'].max(), df_train.loc[df_train['length_question1'] == 146].question1.item(), df_train.loc[df_train['length_question2'] == 271].question2.iloc[0]))
#

# scatter plot to see the relation of both questions length in each axis

sns.jointplot(size=20, ratio=5, x="length_question1", y="length_question2", data=df_train[['length_question1', 'length_question2']]);
#

# scatter plot for both variables, color by is_duplicate or not

sns.lmplot(x="length_question1", y="length_question2", data=df_train[['length_question1', 'length_question2', 'is_duplicate']],

           fit_reg=False, hue='is_duplicate', legend=True, size=15)

#

# Move the legend to an empty part of the plot

plt.legend(loc='lower right')
#

# caluclate the difference in between lengths for questions (1 and 2)

df_train['length_difference_q12'] = abs(df_train['length_question1'] - df_train['length_question2'])

#

# plot the absolute difference in between the lengths of questions

plt.figure(figsize=(20,10))

sns.regplot(data=df_train, x="length_difference_q12", y="is_duplicate", logistic=True, n_boot=500, y_jitter=.03)

#

# print correlation matrix

df_train[['length_question1', 'length_question2', 'length_difference_q12', 'is_duplicate']].corr()
df_train.head()
from wordcloud import WordCloud

#

# create corpus for both set of questions

question1_corpus = " ".join(df_train['question1'].tolist())

question2_corpus = " ".join(df_train['question2'].tolist())

#

# wordcloud

cloud_1 = WordCloud(width=1920, height=1080, background_color="white", mode="RGB").generate(question1_corpus)

cloud_2 = WordCloud(width=1920, height=1080, background_color="white", mode="RGB").generate(question2_corpus)

# plot definitions

font = {'weight': 'bold', 'size': 28}

plt.figure(figsize=(20, 15))

plt.title("WordCoud for Question 1", loc="center", fontdict=font)

plt.imshow(cloud_1)

plt.axis("off")
# wordcloud for question 2

plt.figure(figsize=(20, 15))

plt.title("WordCoud for Question 2", loc="center", fontdict=font)

plt.imshow(cloud_2)

plt.axis("off")
questions_corpus = question1_corpus + question2_corpus

print("Total naive tokens in corpus {}".format(len(questions_corpus.split(" "))))

cloud = WordCloud(width=1920, height=1080, background_color="white", mode="RGB").generate(questions_corpus)

# plot definitions

font = {'weight': 'bold', 'size': 28}

plt.figure(figsize=(20, 15))

plt.title("WordCoud for Question Corpus", loc="center", fontdict=font)

plt.imshow(cloud)

plt.axis("off")
import re

# import contractions

import nltk

from nltk import word_tokenize

from nltk.util import ngrams

from collections import Counter

#

# pre-process text to fix contractions and remove question marks and non alphanumeric characters

# to not sum them in the grams as corpus was joint into one

# questions_corpus = contractions.fix(questions_corpus)

questions_corpus = re.sub(r"'", '',questions_corpus)

questions_corpus = re.sub(r'[^A-Za-z0-9]', ' ',questions_corpus)

# tokenize resulting corpus by words

token = nltk.word_tokenize(questions_corpus)

# build ngrams for 3, 4 and 5 windows

trigrams = ngrams(token,3)

fourgrams = ngrams(token,4)

fivegrams = ngrams(token,5)

trigram_counter = Counter(trigrams)

fourgram_counter = Counter(fourgrams)

fivegram_counter = Counter(fivegrams)
tgc_t10 = trigram_counter.most_common(10)

fgc_t10 = fourgram_counter.most_common(10)

ftgc_10 = fivegram_counter.most_common(10)

print("------ TRIGRAM ------")

for idx in range(0, len(tgc_t10)):

    print("Word Combination {0}, frequency: {1}".format(tgc_t10[idx][0],tgc_t10[idx][1]))

print("------ FOURGRAM ------")

for idx in range(0, len(fgc_t10)):

    print("Word Combination {0}, frequency: {1}".format(fgc_t10[idx][0],fgc_t10[idx][1]))

print("------ FIVEGRAM ------")

for idx in range(0, len(ftgc_10)):

    print("Word Combination {0}, frequency: {1}".format(ftgc_10[idx][0],ftgc_10[idx][1]))