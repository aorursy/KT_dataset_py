import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import os

print(os.listdir("../input"))

dataset = pd.read_csv('../input/restaurantreviews/Restaurant_Reviews.tsv', delimiter='\t', quoting = 3) #3 means ignoring double quotes 
dataset.head()
#Impoorting required packages

import re

#dopwnloading Stopwords list

import nltk

nltk.download('stopwords')

#importing stopwords

from nltk.corpus import stopwords

#importing stemming package

from nltk.stem.porter import PorterStemmer
#Cleaning the texts

#Removing puncutations, and number

#^ represents don't want to remove

#Keeping the letters a to z and A to Z with space

#collection of text is called as corpus

#Creating the Bag of words model

corpus = []

for i in range(0,1000):

    

    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])

    

    #putting all letters in lower case

    review = review.lower()

    

    #Removing stopwords from string and stemming the word

    

    #Stemming is used for make the words to normal form (root) like loved will become like love, loving will become love and capital letter of first letter will become small

    #Stemming is taking of root of the word

    

    review = review.split()

    

    ps = PorterStemmer()

    

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    

    #Joining the words to make a string

    

    review =' '.join(review)

    corpus.append(review)
corpus
#Tokenization is the process of taking all words of the review and making them one column for each word

from sklearn.feature_extraction.text import CountVectorizer

#max_features is used to remove non relavent words

cv = CountVectorizer(max_features = 1500)

#Spars metrics in NLP

X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1].values
#Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#Naive bayes, decission tree, and random forest common models for NLP

#Fitting naive Bayes to the train set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
#Predictions

y_pred = classifier.predict(X_test)
#Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
#Accuracy

(55+91)/200
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=cm)

plt.show()
fig, ax = plot_confusion_matrix(conf_mat=cm,show_absolute=True,show_normed=True,colorbar=True)

plt.show()
#Naive bayes, decission tree, and random forest common models for NLP

#Fitting Decission tree to the train set

from sklearn import tree

dtree = tree.DecisionTreeClassifier()

dtree.fit(X_train, y_train)
#Predictions

y_pred_dtree = dtree.predict(X_test)
cm_dtree = confusion_matrix(y_test, y_pred_dtree)
fig, ax = plot_confusion_matrix(conf_mat=cm_dtree)

plt.show()
#Accuracy

(72+63)/200