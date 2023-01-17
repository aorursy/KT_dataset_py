import numpy as np 

import pandas as pd 

import nltk

from nltk.corpus import stopwords

import string

import os

import re

import matplotlib.pyplot as plt
os.getcwd()
os.chdir("../input/spam-ham-emails")
df = pd.read_csv("emails.csv") #read the CSV file
df.head(5)
df.shape
df.spam.value_counts()
df.columns
df.drop_duplicates(inplace = True)
df.shape
df.isnull().sum()
from string import punctuation

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import SnowballStemmer
def clean_txt(sent):

    #Stripping white spaces before and after the text

    sent = sent.strip()

    #Replacing multiple spaces with a single space

    result = re.sub("\s+", " ", sent)

    #Replacing Non-Alpha-numeric and non space charecters with nothing

    result1 = re.sub("[^\w\s]","",result)

    tokens = word_tokenize(sent.lower())

    stop_updated = stopwords.words("english")  +  ["would", "could","told","subject"]

    text = [term for term in tokens if term not in stop_updated and len(term) > 2] 

    res = " ".join(text)

    return res
df['text'] = df.text.apply(clean_txt)
df.head()
#Seperate text column and the labels into X and y

X_text = df.text.values

y = df.spam.values
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=3500)

X = tfidf_vectorizer.fit_transform(X_text)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)