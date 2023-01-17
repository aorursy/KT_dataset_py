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
#Read data from a tab separeted file, using a delimeter '\t' for separating feature values.
dataset = pd.read_csv('../input/amazon_alexa.tsv',delimiter='\t',quoting=3)
#Reviewing data
dataset.head(5)
#Using nltk library for preparing data for ready to use for sentimental analysis
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
reviews=[]
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]',' ',dataset['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    reviews.append(review)
#Creating a bag of words that contains all the unique words in all the reviews
#Each column in X will represent each word
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(reviews).toarray()
y = dataset.iloc[:,4]
#Split data into train and test set
from sklearn.model_selection import train_test_split
xtr, xtst, ytr, ytst = train_test_split(X,y,test_size=0.25,random_state=0)
#Building a Naive bayes classifier for classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

#fitting train data in the classifier
classifier.fit(xtr,ytr)
classifier.score(xtr,ytr)
#Testing our classifier onto test data and storing the results in y_pred variable
y_pred = classifier.predict(xtst)
#Checking accuracy of predictions on test data
#We are using confusion matrix from sklearn.metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytst,y_pred)
print((cm[0,0]+cm[1,1])/cm.sum()*100)
#Modelling a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier2 = DecisionTreeClassifier(criterion='entropy',random_state=0)

#Training the classifier
classifier2.fit(xtr,ytr)
classifier2.score(xtr,ytr)
#Prediction over test data
y_pred2 = classifier2.predict(xtst)
#Check accuracy
cm = confusion_matrix(ytst,y_pred2)
print((cm[0,0]+cm[1,1])/cm.sum()*100)
