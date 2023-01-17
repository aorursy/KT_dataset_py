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
import matplotlib.pyplot as plt

import math

import random
#loading the dataset

dataset = pd.read_csv("../input/Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)
print(dataset.info())
print(dataset.head())
print(dataset.describe())
#cleaning the texts

#importing re as its is most used library to clean text.

import re

review = re.sub("[^a-zA-Z]", " ",  dataset["Review"][0])

#here re.sub sort the text on by the letter from A to Z and get free of all the other punctuation, nubers, etc.

"""here our input to the sub command is "[^a-zA-Z]" which means we want everything to remove or sort expect the letter

from a to z and also capital letters from A to Z, also here zA is not seperated and this create a problem and

so to sort that we add a new paramer space """
print(review)
# the next process will be to convert all the letter to a lower case letter

review = review.lower()

print(review)
# now in the next process we will get rid of word which are of no use to us to judge the review like article, prepositions, etc.

#we need a library which contain most functions of NLP calles nltk

import nltk



""" now we will run a for loop for all the review for the same process we done above for 1st review"""

review =review.split()

print(review)
from nltk.corpus import stopwords

review = [word for word in review if not word in set(stopwords.words("english"))]

#the set function is used to speedup the algo as it guide fot the set of words in the list

print(review)
#now appling stemming. we need to import porterstemmer

from nltk.stem.porter import PorterStemmer

#we will redesign our algorithem as the above process was to make understanding and now we can have a fun algorithem here.

ps = PorterStemmer()

review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]

print(review)
#we will make a list containg all the words and so we use join command with space as to seperate the words for join in single word.

review = " ".join(review)

print(review)
#the same process we will perform for all 1000 review.

#we will make a list for all 1000 reviews names corpus

corpus = []

for i in range(0, 1000):

    review_1 = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    review_1 = review_1.lower()

    review_1 = review_1.split()

    ps = PorterStemmer()

    review_1 = [ps.stem(word) for word in review_1 if not word in set(stopwords.words('english'))]

    review_1 = ' '.join(review_1)

    corpus.append(review_1)
corpus[:10]
#creating a bag of word model, appluing the machine learning model and tranforming and predicting.

""" we will make a table containing all the review in one column. And then we choose a specific word and count the 

number of times the specific word appeared in the review """

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

#we can do use stemming , sorting and many other things in countvectorizer but can be more efficient doing it manually.

X = cv.fit_transform(corpus).toarray()
print(X[:10])
print(X.shape)
#here we can see that the 1565 words are taken from the reviews and so we add a new parameter in countVectorizer a max_feature.

cv = CountVectorizer(max_features = 1500)

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1].values

print(X.shape)
""" we will train our model to predict of the review. The most common model used in Natual Language Processing are

naive bayes decision tree and random forest. We will use here naive bayes first""" 

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)



# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report

target =["negative add", "positive add"]

print(classification_report(y_test, y_pred, target_names=target))
a = (48+86)/200

print("accuracy {}\n".format(a))
#Now we will train random forest to the dataset

from sklearn.ensemble import RandomForestClassifier

SEED = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=0.12,max_depth = 5,  random_state=SEED)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test, y_pred_rf)

print(cm_rf)

from sklearn.metrics import classification_report

target =["negative add", "positive add"]

print(classification_report(y_test, y_pred_rf, target_names=target))