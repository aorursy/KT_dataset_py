import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

import nltk
#nltk.download()
from nltk.corpus import stopwords
import string

from scipy.sparse import hstack

import matplotlib.pyplot as plt
import seaborn as sns
import random

%matplotlib inline
# controls whether we work with only a subset of the data
SAMPLE_DATA = None
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
train.head()
train.describe()
sns.countplot(x='target',data=train)
train_sample = train.sample(n=50)
train_sample.question_text.head(n=25)
train[train.target == 1].question_text
# temporarily calculate metafeatures against only train_sample to speed up iterative development
if SAMPLE_DATA:
    train = train_sample
from nltk import word_tokenize
%timeit train['question_tokens'] = train['question_text'].str.lower().str.split(' ')
#train['question_tokens'].head()
train['question_length'] = train['question_text'].str.len()
train.head()
train['question_tokens_length'] = train['question_tokens'].str.len()
train.head()
eng_stopwords = set(stopwords.words("english"))
random.sample(eng_stopwords,15)
def num_stopwords(question_tokens):
    return len([w for w in question_tokens if w in eng_stopwords])
train['question_num_stopwords'] = train['question_tokens'].apply(lambda question_tokens: num_stopwords(question_tokens))
train.head()
def num_punctuation(question_text):
    punctuation_marks = list(string.punctuation)
    return len([t for t in list(question_text) if t in punctuation_marks])
train['num_punctuation'] = train['question_text'].apply(lambda question_text:num_punctuation(question_text))
train.head()
train['percent_punctuation'] = train['num_punctuation'] / train['question_length']
train['percent_stopwords'] =train['question_num_stopwords']/train['question_tokens_length']
train.head()
def mean_word_length(row):
    return np.mean([len(str(w)) for w in row['question_tokens']])

train['mean_word_length'] = train['question_tokens'].apply(lambda question_tokens:np.mean([len(w) for w in question_tokens]))
train.head()
def get_pos(question_tokens):
    pos_list = nltk.pos_tag([w for w in question_tokens if w])
    return [pos[1] for pos in pos_list]
train['pos'] = train['question_tokens'].apply(lambda question_tokens:get_pos(question_tokens))
train.head()
from statistics import mode
import statistics

def get_mode_of_pos(pos):
    if not pos:
        return ''
    else:
        poses = [p for p in pos if isinstance(p, str)]
        if len(poses):
            try:
                m = mode(poses)
                return m
            except statistics.StatisticsError as e:
                return ''
train['pos_mode'] = train['pos'].apply(lambda pos:get_mode_of_pos(pos))
train.head()
train['pos_first'] = train['pos'].str.slice(0,1)
train['pos_last'] = train['pos'].str.slice(-1,1)
#train.head()
sns.boxplot(x='target',y="percent_stopwords",data=train)
sns.boxplot(x='target',y="question_length",data=train)
sns.boxplot(x='target',y="percent_punctuation",data=train)
sns.boxplot(x='target',y="question_tokens_length",data=train)
sns.boxplot(x='target',y="mean_word_length",data=train)
sns.boxplot(x='target',y="mean_word_length",hue="pos_mode",data=train)
sns.catplot(x='target',y="pos_mode",data=train)
train.hist(figsize=(15,20))
plt.figure()
train.to_csv('train_preprocessed.csv')