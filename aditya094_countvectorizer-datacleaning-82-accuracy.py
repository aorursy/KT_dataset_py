# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords # for stopwords
import string 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv") 
train_data.head()
train_data["author"].unique()
train_data["author"].value_counts()
# we don't need id for now, id has no effect on output
train_data.drop( "id" , axis=1 , inplace = True)
# Let's seperate the documents and output(author name) 

train_document = train_data["text"]
train_authors = train_data["author"]

#for testing data

test_document = test_data["text"]


#Let's create stopword list 

stopword_list = stopwords.words('english')
stopword_list[0:5]
#let's create new list of punctuation which should be removed from documents

punct = list(string.punctuation)
punct[0:5]
stopword_list = stopword_list + punct
from sklearn.feature_extraction.text import CountVectorizer #for converting documents into matrix form
from sklearn.model_selection import train_test_split # for splitting the training data
# Let's split the training data into train and test data
x_train , x_test , y_train , y_test = train_test_split ( train_document , train_authors)
cv = CountVectorizer(stop_words = stopword_list)
xtrain = cv.fit_transform( x_train ) # train the model and find features using training data
xtest = cv.transform( x_test )
#Let's import classifiers 
from sklearn.naive_bayes import MultinomialNB 
clf = MultinomialNB()
clf.fit(xtrain,y_train)
clf.score(xtest , y_test)
# a gud accuracy we found using multinomial naive bayes
training_data = cv.fit_transform(train_document)
testing_data = cv.transform(test_document)
#It's time to predict the output
clf.fit(training_data ,train_authors)
prediction = clf.predict(testing_data)