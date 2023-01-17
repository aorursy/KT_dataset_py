# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Create dataframe with just the title, description, and genre list

df = pd.read_csv("../input/rotten-tomatoes-movies-and-critics-datasets/rotten_tomatoes_movies.csv", na_filter=False)

df = df[{"movie_title","genre","movie_info"}]



#drop blank rows

df= df[df['genre'] != ''].reset_index(drop = True) 



df.tail(8)

import re

import nltk

from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords



# Load stopwords, common words such as  "a," "the," "it," etc.

stop_words = stopwords.words('english')

    

#Initialize stemmer, which will take words and convert words to their "stem," e.g. Playing-> Play

ps = PorterStemmer() 



# Removes non-alphabetical characters, whitespaces, and converts all letters to lowercase

# References: https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/

def clean_text(txt): 

    txt= txt.lower()   #lowercase

    txt= re.sub("[^a-zA-Z]"," ",txt) #Remove everything except alphabetical characters 

    txt= word_tokenize(txt) #tokenize (split into list and remove whitespace)

    

    #initialize list to store clean text

    clean_text=""

    

    #iterate over each word

    for w in txt:      

        #remove stopwords

        if w not in stop_words:

            #stem=ps.stem(w) #stem 

            stem=w

            clean_text += stem +" " 

    return clean_text





movie_info_new=[] #declare a list to hold new movies



for cell in df['movie_info']:    

    txt= clean_text(cell)

    movie_info_new.append(txt)

    

#add new info column to the dataframe

df['movie_info_new'] = movie_info_new 

df.tail()
from sklearn.preprocessing import MultiLabelBinarizer

import json



## format genre column to be in the form of a list rather than string



genre_new=[] #declare a list

for cell in df['genre']:

    cell=cell.replace(" ", "") #remove whitespace

    cell=cell.replace("&", "& ") #add whitespace back in for ampersands

    genre_new.append(cell.split(",")) #for each genre cell, create a list of items from the original string, using a comma as a delimeter

    

#add new genre column to the dataframe

df['genre_new'] = genre_new 



## MultiLabelBinarizer takes an iterable list and turns it into columns with binary values that represent the list.

## For example, [Comedy, Drama] -> Comedy and Drama columns with a value of 1, all other columns with a value of 0



#initialize MultiLabelBinarizer 

mlb = MultiLabelBinarizer() 



#transform the genre_new column to a series of columns with binary values

binary_labels=pd.DataFrame(mlb.fit_transform(df['genre_new']),columns=mlb.classes_) 



#order columns alphabetically

binary_labels=binary_labels.sort_index(axis=1) 



binary_labels.tail()



#bring data frames together

movies = df.merge(binary_labels, how='inner', left_index=True, right_index=True)



#Drop non-properly formatted columns

movies= movies.drop(columns=['genre', 'movie_info','genre_new'])



movies.tail(7)
import seaborn as sns

import matplotlib.pyplot as plt

categories = list(binary_labels.columns.values)

ax= sns.barplot(binary_labels.sum().values, categories)



plt.title("Movies for each genre", fontsize=24)

plt.ylabel('Genre', fontsize=18)

plt.xlabel('Number of movies tagged with genre', fontsize=18)

#adding the text labels

rects = ax.patches

labels = binary_labels.sum().values

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split



# split dataset into training and validation set

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

xtrain, xval, ytrain, yval = train_test_split(movies['movie_info_new'], binary_labels, test_size=0.2, random_state=9)



# create TF-IDF features

# TF-IDF = Term frequency - inverse document frequency

# Used to predict how important a word is for a document

# https://en.wikipedia.org/wiki/Tf%E2%80%93idf

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)

xval_tfidf = tfidf_vectorizer.transform(xval)
from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score



#Run Logistic Regrssion

logreg = LogisticRegression()

logreg_classifier = OneVsRestClassifier(logreg)



# fit model on train data

logreg_classifier.fit(xtrain_tfidf, ytrain)



# make predictions for validation set

predictions = logreg_classifier.predict(xval_tfidf)



# evaluate performance

from sklearn.metrics import accuracy_score

print("Accuracy score for Logistic Regression:")

print(accuracy_score(yval, predictions))
from sklearn.metrics import classification_report



#Show precision and recall per genre

print(classification_report(yval, predictions, target_names=binary_labels.columns))
# Using Gaussian Naive Bayes 

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB





# initialize binary relevance multi-label classifier

# with a gaussian naive bayes base classifier

classifier = BinaryRelevance(GaussianNB())

# train

classifier.fit(xtrain_tfidf, ytrain)

# predict

predictions = classifier.predict(xval_tfidf)



print("Accuracy score for Gaussian Naive Bayes:")

print(accuracy_score(yval, predictions))



print("Individual genre predictions:")

print(classification_report(yval, predictions, target_names=binary_labels.columns))
#Make smaller dataset with only the 4 most common genres

movies_02= movies[['movie_info_new', 'Action& Adventure', 'Drama', 'Comedy', 'Mystery& Suspense']]

binary_labels_02=movies[['Action& Adventure', 'Drama', 'Comedy', 'Mystery& Suspense',]]

movies_02.tail(7)

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)



# split dataset into training and validation set

xtrain_02, xval_02, ytrain_02, yval_02 = train_test_split(movies_02['movie_info_new'], binary_labels_02, test_size=0.2, random_state=9)



# create TF-IDF features

xtrain_tfidf_02 = tfidf_vectorizer.fit_transform(xtrain_02)

xval_tfidf_02 = tfidf_vectorizer.transform(xval_02)



#Run Logistic Regression

log_reg_02 = LogisticRegression()

logreg_classifier_02 = OneVsRestClassifier(log_reg_02)



# fit model on train data

logreg_classifier_02.fit(xtrain_tfidf_02, ytrain_02)



# make predictions for validation set

predictions_02 = logreg_classifier_02.predict(xval_tfidf_02)



# evaluate performance

print("Accuracy score for Logistic Regression with only 4 genres:")

print(logreg_classifier_02.score(xval_tfidf_02, yval_02))

print(classification_report(yval_02, predictions_02, target_names=binary_labels_02.columns))
# Using Gaussian Naive Bayes 

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



# initialize binary relevance multi-label classifier

# with a gaussian naive bayes base classifier

classifier = BinaryRelevance(GaussianNB())

# train

classifier.fit(xtrain_tfidf_02, ytrain_02)

# predict

predictions_03 = classifier.predict(xval_tfidf_02)



print("Accuracy score for Naive Bayes with only 4 genres:")

print(accuracy_score(yval_02, predictions_03))

print(classification_report(yval_02, predictions_03, target_names=binary_labels_02.columns))