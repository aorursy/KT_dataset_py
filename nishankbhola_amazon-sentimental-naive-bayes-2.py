



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bz2

import gc

import chardet

import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train_file = bz2.BZ2File('../input/amazonreviews/train.ft.txt.bz2')

test_file = bz2.BZ2File('../input/amazonreviews/test.ft.txt.bz2')







#reading the data and appending it to python-list

train_file_lines = train_file.readlines()

test_file_lines = test_file.readlines()

del train_file, test_file

train_file_lines = [x.decode('utf-8') for x in train_file_lines]

test_file_lines = [x.decode('utf-8') for x in test_file_lines]



#Seprating scores from reviews

length_of_train = len(train_file_lines)

length_of_test = len(test_file_lines)

train_file_lines_score = []

test_file_lines_score = []

for i in range(0,length_of_train):

    temp = train_file_lines[i][9]

    train_file_lines_score.append(temp)



for i in range(0,length_of_test):

    temp = test_file_lines[i][9]

    test_file_lines_score.append(temp)

print(length_of_train , length_of_test , len(train_file_lines_score) , len(test_file_lines_score))
#Converting python list to dataframe

df_train_reviews = pd.DataFrame({'reviews': train_file_lines} )

df_test_reviews = pd.DataFrame({'reviews': test_file_lines} )

df_train_score = pd.DataFrame({'score': train_file_lines_score} )

df_test_score = pd.DataFrame({'score': test_file_lines_score} )



#taking less reviews due to limited computational power

df_train_reviews = df_train_reviews[:100000]

df_test_reviews = df_test_reviews[:25000]

df_train_score = df_train_score[:100000]

df_test_score = df_test_score[:25000]





print(df_train_reviews.shape,df_test_reviews.shape,df_train_score.shape,df_test_score.shape)
#pre-processing data

import re

import nltk

from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer



#Pre-processing reviews data of traning set

corpus_train = []

for i in range(0, len(df_train_reviews)):

    review = re.sub("<.*?>", "", df_train_reviews['reviews'][i])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    del review[0]

    #ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer() 

    #review = [ps.stem(words) for words in review if not words in set(stopwords.words('english'))]

    review = [lemmatizer.lemmatize(words) for words in review if not words in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_train.append(review)

    

#Pre-processing reviews data of testing set

corpus_test = []

for i in range(0, len(df_test_reviews)):

    review = re.sub("<.*?>", "", df_test_reviews['reviews'][i])

    review = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    review = review.split()

    del review[0]

    #ps = PorterStemmer()

    lemmatizer = WordNetLemmatizer() 

    #review = [ps.stem(words) for words in review if not words in set(stopwords.words('english'))]

    review = [lemmatizer.lemmatize(words) for words in review if not words in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus_test.append(review)



corpus_train[0]





#Bag of words(sparc matrix)

from sklearn.feature_extraction.text import TfidfVectorizer

#cv = TfidfVectorizer(ngram_range=(1,2))

cv = TfidfVectorizer(max_features=9000)

X_train = cv.fit_transform(corpus_train).toarray()

X_test = cv.transform(corpus_test).toarray()

X_train.shape



# from sklearn.preprocessing import StandardScaler

# scaler1 = StandardScaler(with_mean=False)

# scaler = scaler1.fit(X_train)

# X_train= scaler.transform(X_train)

# X_test = scaler.transform(X_test)
#Naive bayes classification

from sklearn.naive_bayes import MultinomialNB



classifier = MultinomialNB(alpha = 1 , fit_prior=True, class_prior=None)

classifier.fit(X_train, df_train_score)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

def accuracy(confusion_matrix):

    diagonal_sum = confusion_matrix.trace()

    sum_of_all_elements = confusion_matrix.sum()

    return diagonal_sum / sum_of_all_elements 



#making the confusion matrix

cm = confusion_matrix(df_test_score , y_pred)

accuracy(cm)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(df_test_score , y_pred)

accuracy