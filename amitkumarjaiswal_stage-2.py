import csv

import json

import os

import string

import sys

import warnings



import matplotlib

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import stopwords

import numpy as np

import pandas as pd

import scipy

import seaborn as sns

import sklearn



import gensim
# Visualization setting

sns.set(style='white', context='notebook', palette='deep')

warnings.filterwarnings('ignore')

sns.set_style('white')
# load data

df = pd.read_csv('../input/test_stage_2.tsv', delimiter='\t')

submit = pd.read_csv('../input/sample_submission_stage_2.csv')
def check_missing_data(df):

    flag = df.isna().sum().any()

    if flag:

        total = df.isnull().sum()

        prop = (df.isnull().sum()) / (df.isnull().count()*100)

        output = pd.concat([total, prop], axis=1, keys=['Total', 'Percent'])

        data_type = []

        for col in df.columns:

            dtype = str(df[col].dtype)

            data_type.append(dtype)

        output['Types'] = data_type

        return(np.transpose(output))

    else:

        return(False)

check_missing_data(df)
# Extra features

df['n_words'] = df['Text'].apply(lambda x: len(str(x).split()))

df['unq_words'] = df['Text'].apply(lambda x: len(set(str(x).split())))

stop_words = set(stopwords.words('english'))

df['n_stopwords'] = df['Text'].apply(lambda x: \

    len([word for word in str(x).lower().split() if word in stop_words]))

df['n_punctuations'] = df['Text'].apply(lambda x: \

    len([c for c in str(x) if c in string.punctuation])) 

df['n_words_upper'] = df['Text'].apply(lambda x: \

    len([word for word in str(x).split() if word.isupper()]))



df.to_csv('train.tsv', sep='\t')