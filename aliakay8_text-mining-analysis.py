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
import numpy as np
import re
import pickle 
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')
# Unpickling dataset
X_in = open('../input/uX.pickle','rb')
y_in = open('../input/uy.pickle','rb')
X = pickle.load(X_in)
y = pickle.load(y_in)
type(X), len(X)
X[10]
y
# Creating the corpus
corpus = []
for i in range(0, 2000):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)  
corpus[5]
##Bag of Words
# Creating the BOW model
# min_df = 3      a word should pass at least 3 doc
# max_df = 0.6    words that pass in more than %60 of words is also eliminated

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
X.shape
X
# Creating the Tf-Idf model directly
##The Math
##TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
##IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
##Value = TF * IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

text_train.shape
# Training the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)
# Testing model performance
sent_pred = classifier.predict(text_test)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(sent_test, sent_pred)
cm
accuracy_score(sent_test, sent_pred)
# Saving our classifier
with open('myclassifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# Saving the Tf-Idf model
with open('mytfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
# Using our classifier
with open('mytfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
    
with open('myclassifier.pickle','rb') as f:
    clf = pickle.load(f)
    
sample = ["You are a bad person man, go to hell"]
sample = tfidf.transform(sample).toarray()
sentiment = clf.predict(sample)
sentiment
sample = ["You are a nice person man, have a good life"]
sample = tfidf.transform(sample).toarray()
sentiment = clf.predict(sample)
sentiment
