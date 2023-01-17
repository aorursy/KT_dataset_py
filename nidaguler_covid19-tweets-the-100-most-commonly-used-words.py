# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

import nltk as nlp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv(r"../input/covid19-tweets/covid19_tweets.csv")

df=pd.concat([df.source,df.text],axis=1)

df.head()
df.dropna(axis=0,inplace=True)
text_list=[]

for text in df.text:

    text=re.sub("[^a-zA-Z]"," ",text)

    text=text.lower()

    text=nltk.word_tokenize(text)

    lemma  = nlp.WordNetLemmatizer()

    text=[lemma.lemmatize(word) for word in text]

    text=" ".join(text)

    text_list.append(text)
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features =100

count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(text_list).toarray()

print("The 100 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))