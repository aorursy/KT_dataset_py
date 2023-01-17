import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#from KaggleWord2VecUtility import KaggleWOrd2VecUtility
from pandas import Series, DataFrame
#from corpora import Corpus
import pandas as pd
import numpy as np
#import urllib2
import json
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
print (0)
#importing the train and test data sets
train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")

print (0)
#use the Natural Language Tool Kit word lemmatizer to stem the words in the ingredients lists
#stemmer  = WordNetLemmatizer()
#Do I need to download this package since I'm using Kaggle's site?
#nltk.download('wordnet')
#ingredients = train_df["ingredients"]

#corpus = stemmer.lemmatize(ingredients)
#train_df['ingredients_clean'] = [' , '.join(z).strip() for z in train_df['ingredients']]

#test_df['ingredients_clean'] = test_df['ingredients']

print (0)
train_df['ingredients_clean'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', '', line)) 
                                         for line in lists]).strip() for lists in train_df['ingredients']]

test_df['ingredients_clean'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', '', line)) 
                                         for line in lists]).strip() for lists in test_df['ingredients']]

print (0)
corpus_train = train_df['ingredients_clean']
corpus_test = test_df['ingredients_clean']
print (0)
vectorizer_train = TfidfVectorizer(stop_words = 'english', ngram_range = (1,1), analyzer = "word",
                                    max_df = 1.0, token_pattern = r'\w+')
vectorizer_test = TfidfVectorizer(stop_words = 'english')

tfidf_train = vectorizer_train.fit_transform(corpus_train)
#tfidf2 = vectorizer1.fit_transform(corpus_test)
predictor = tfidf_train
x = predictor
print (0)
target = train_df['cuisine']
y = target
print (0)
#Logistic Regression model
parameter = {'C':[1, 80000]}

LR = LogisticRegression()

print(0)
classifier = grid_search.GridSearchCV(LR, parameter)
classifier = classifier.fit(x, y)

print(0)
prediction = pd.DataFrame(classifier.predict(predictor))
test_df['cuisine'] = ''
test_df['cuisine'] = prediction

#type(test_df['cuisine'])
#print (prediction)
#type(prediction)
#prediction.__len__()
#print (prediction)
#x = df.columns.to_series().groupby(df.dtypes).groups
print (0)
submission_python = test_df[['id', 'cuisine']]

FALSE = 0
TRUE = 1

submission_python.to_csv("LR_Submission.csv", index = FALSE)

print (0)