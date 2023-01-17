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
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix ,accuracy_score

###Importig dataset
dataset = pd.read_csv('../input/amazon_alexa.tsv', delimiter = '\t', quoting = 3)
dataset.head(3)
print("The Shape of dataset", dataset.shape)
corpus = []
for i in range(0, 3150):
    # column : "verified_reviews", row ith 
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i])
    # convert all cases to lower cases 
    review = review.lower()
    # split to array(default delimiter is " ") 
    review = review.split()
    # creating PorterStemmer object to 
    # take main stem of each word
    ps = PorterStemmer()
    # loop for stemming each word 
    # in string array at ith row 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)
    # append each string to create 
    # array of clean text  
    corpus.append(review)
    
corpus[2]
dataset['verified_reviews'][2]
###Creating the bag of words model
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 4].values
##Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

###Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)
###predicting the test set results
y_pred = classifier.predict(X_test)
y_pred
##Making the Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm
##Chcek accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
#Modelling a Decision Tree Classifier
classifier2 = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier2.fit(X_train, y_train)
##Predict model 
y_pred1 = classifier2.predict(X_test)
y_pred1
##making confusion matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm1
##Chcek accuracy score
accuracy_score(y_test, y_pred1)
# Joinining all the reviews into single paragraph 
rev_string = " ".join(corpus)
##rev_string
####Draw wordcloud for all reviews
wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(rev_string)

plt.imshow(wordcloud_ip)
