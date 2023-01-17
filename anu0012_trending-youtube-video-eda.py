# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk
from nltk import word_tokenize, ngrams
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import xgboost as xgb
np.random.seed(25)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
ca_data = pd.read_csv("../input/CAvideos.csv")
de_data = pd.read_csv("../input/DEvideos.csv")
fr_data = pd.read_csv("../input/FRvideos.csv")
US_data = pd.read_csv("../input/USvideos.csv")
ca_data.head()
ca_data.isnull().sum(axis=0)
ca_data['description'].fillna("Not available", inplace=True)
ca_data.dtypes
col = ['category_id', 'views', 'likes', 'dislikes','comment_count']

ca_data[col].describe()
ca_data.loc[ca_data['views'].idxmax()]
ca_data.loc[ca_data['views'].idxmin()]
ca_data.loc[ca_data['comment_count'].idxmax()]
ca_data['channel_title'].value_counts().head(10).plot.barh()
ca_data['channel_title'].value_counts().tail(10).plot.barh()
ca_data['category_id'].value_counts().head(10).plot.barh()
ca_data['category_id'].value_counts().tail(10).plot.barh()
