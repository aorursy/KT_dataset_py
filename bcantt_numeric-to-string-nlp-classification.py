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
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
for index, row in data.iterrows():

    data.loc[index, 'the_last_word'] = str(row['SepalLengthCm']) + " " + str(row['SepalLengthCm'] + row['SepalWidthCm']) + " " + str(row['SepalLengthCm'] + row['PetalLengthCm']) +  " " + str(row['SepalLengthCm'] + row['PetalWidthCm']) + " " + str(row['SepalLengthCm'] - row['SepalWidthCm']) + " " + str(row['SepalLengthCm'] - row['PetalLengthCm']) + " " + str(row['SepalLengthCm'] - row['PetalWidthCm']) +" "+ str(row['SepalWidthCm']) +" " + str(row['SepalWidthCm'] + row['PetalLengthCm']) + " " + str(row['SepalWidthCm'] + row['PetalWidthCm']) + " " + str(row['SepalWidthCm'] - row['PetalLengthCm']) + " " + str(row['SepalWidthCm'] - row['PetalWidthCm']) + str(row['PetalLengthCm']) +" " + str(row['PetalLengthCm'] + row['PetalWidthCm']) + " "+ str(row['PetalLengthCm'] - row['PetalWidthCm']) 



                                         
data
last_words = data['the_last_word'].values

y = data['Species'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(last_words, y, test_size=0.20, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(binary=True, use_idf=True)

tfidf_train_data = vec.fit_transform(X_train)

tfidf_test_data = vec.transform(X_test)

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()

classifier.fit(tfidf_train_data, y_train)

score = classifier.score(tfidf_test_data, y_test)



print("Accuracy:", score)