# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import json

import nltk

import re

import csv

import matplotlib.pyplot as plt 

import seaborn as sns

from tqdm import tqdm

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

import spacy

from sklearn.preprocessing import LabelEncoder



%matplotlib inline

pd.set_option('display.max_colwidth', 300)
df = pd.read_csv(r'/kaggle/input/kollywood-movie-dataset-2011-2017/Kollywood Movie Dataset (2011 - 2017).csv')

df.sample(5)
df.shape
len(df.Genre.unique())
df["Genre"] = df["Genre"].apply(lambda x: x.split()[0])

df["Genre"] = df["Genre"].apply(lambda x: x.replace(r"/", ""))

df["Genre"] = df["Genre"].apply(lambda x: x.replace("—", " "))

df["Genre"] = df["Genre"].apply(lambda x: x.replace("-", " "))

df["Genre"] = df["Genre"].apply(lambda x: x.split()[0])

df["Genre"] = df["Genre"].apply(lambda x: x.replace(r",", ""))

df["Genre"] = df["Genre"].apply(lambda x: x.replace(r"romanctic", "romantic"))

df["Genre"] = df["Genre"].apply(lambda x: x.replace(r"sci", "scifi"))

df["Genre"] = df["Genre"].apply(lambda x: x.replace(r"romance", "romantic"))

df.sample(5)
df = df[~df["Genre"].isin(['scifience', 'mob','unknown','road'])]

df.shape
df.Genre.unique()
df.Genre.value_counts()
df.Genre.value_counts().idxmax()
figd=(15,7)

fig,ax=plt.subplots(figsize=figd)

sns.countplot(x = "Genre", data=df, order = df.Genre.value_counts().index)

plt.xticks(rotation=90)

plt.show()
figd=(8,5)

fig,ax=plt.subplots(figsize=figd)

sns.countplot(x = "Release Year", data=df, order = df["Release Year"].value_counts().index)

plt.xticks(rotation=30)

plt.show()
df.groupby("Genre")["Rating"].max().sort_values(ascending=False)
figd=(15,7)

fig,ax=plt.subplots(figsize=figd)

m = df.groupby("Genre")["Rating"].max().sort_values(ascending=False)

sns.barplot(x = m.index, y= m.values, data=df, ci=None)

plt.xticks(rotation=90)

plt.show()
nlp = spacy.load('en')
def normalize(msg):

    

    doc = nlp(msg)

    res=[]

    

    for token in doc:

        if(token.is_stop or token.is_digit or token.is_punct or not(token.is_oov)):

            pass

        else:

            res.append(token.lemma_.lower())

    

    return " ".join(res)
df['Clean_plot'] = df['Plot'].apply(lambda x: normalize(x))

df.sample(5)
def freq_words(x, terms = 30): 

  all_words = str(x).split() 

  fdist = nltk.FreqDist(all_words) 

  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 

  

  # selecting top 20 most frequent words 

  d = words_df.nlargest(columns="count", n = terms) 

  

  # visualize words and frequencies

  plt.figure(figsize=(12,15)) 

  ax = sns.barplot(data=d, x= "count", y = "word") 

  ax.set(ylabel = 'Word') 

  plt.show()
# print 50 most frequent words 

freq_words(df['Clean_plot'], 50)
le = LabelEncoder()

df["Genre"] = le.fit_transform(df["Genre"])

df.sample(5)
tfidf_vectorizer = TfidfVectorizer()

def feature_extraction(msg):

    

    mat = pd.DataFrame(tfidf_vectorizer.fit_transform(msg).toarray(),columns=tfidf_vectorizer.get_feature_names(),index=None)

    return mat
train_x,train_y, test_x,test_y = train_test_split(feature_extraction(df['Clean_plot']),df['Genre'], test_size=0.3)
train_x.head()
test_x.head()
clfs = {

    'mnb': MultinomialNB(),

    'gnb': GaussianNB(),

    'mlp1': MLPClassifier(),

    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),

    'ada': AdaBoostClassifier(),

    'dtc': DecisionTreeClassifier(),

    'rfc': RandomForestClassifier(),

    'gbc': GradientBoostingClassifier(),

    'lr': LogisticRegression()

}
f1_scores = dict()

for clf_name in clfs:

    clf = clfs[clf_name]

    clf.fit(train_x, test_x)

    y_pred = clf.predict(train_y)

    f1_scores[clf_name] = accuracy_score(y_pred, test_y)

    print(clf,":", f1_scores[clf_name])