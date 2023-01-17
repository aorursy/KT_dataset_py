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
traindata=pd.read_csv("../input/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
traindata.review.head()
traindata.review[0]
traindata.review[1]
## to remove html markups we use beautiful soup

from bs4 import BeautifulSoup

example=BeautifulSoup(traindata.review[0])
example
example.get_text()
traindata.review[0]
#Dealing with punctuations,stopwords etc
import re

re.sub("[^a-zA-Z]"," ",example.get_text())
# all backslash and forward slash values are removed

# ^ means not present in 
text1=re.sub("[^a-zA-Z]"," ",example.get_text())
text1=text1.lower()    # important preprocessing steps

text1
# remove stopwords

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
stpwords=stopwords.words('english')
stpwords
#Removing stopwords
#code to work on training examples
def review_to_words(raw_review):

    review_text=BeautifulSoup(raw_review).get_text()

    regex_review=re.sub("[^a-zA-Z]"," ",review_text)

    lower_review=regex_review.lower().split()

    list_review=[w for w in lower_review if w not in stpwords]

    return " ".join(list_review)

%time

clean_review=review_to_words(traindata.review[0])
def review_to_words(raw_review):

    review_text=BeautifulSoup(raw_review).get_text()

    regex_review=re.sub("[^a-zA-Z]"," ",review_text)

    lower_review=regex_review.lower().split()

    stpwords=set(stopwords.words('english'))

    list_review=[w for w in lower_review if w not in stpwords]

    return " ".join(list_review)

%time

clean_review=review_to_words(traindata.review[0])
traindata.review[0]
#Two elements here are new: First, we converted the stop word list to a different data type, 

#a set. This is for speed; since we'll be calling this function tens of thousands of times, 

# it needs to be fast, and searching sets in Python is much faster than searching lists.
num_reviews=len(traindata)
num_reviews
cleaned_train_reviews=[]

for i in range(num_reviews):

    cleaned_train_reviews.append(review_to_words(traindata.review[i]))
cleaned_train_reviews[3]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer='word',max_features=8000,ngram_range=(1,2))
traindata_features=vectorizer.fit_transform(cleaned_train_reviews)
train_data_features=traindata_features.toarray()
train_data_features.shape
df=pd.DataFrame(data=train_data_features,columns=vectorizer.get_feature_names())
df.head()
df.iloc[0][df.iloc[0] !=0]
cleaned_train_reviews[0]
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=200)

forest=forest.fit(train_data_features,traindata.sentiment)
test=pd.read_csv("../input/testData.tsv",header=0,delimiter='\t',quoting=3)

test.head()
num_test=len(test)
cleaned_test_Reviews=[]

for i in range(num_test):

    cleaned_test_Reviews.append(review_to_words(test.review[i]))

    
test_data_features=vectorizer.transform(cleaned_test_Reviews)
test_df=pd.DataFrame(data=test_data_features.toarray(),columns=[vectorizer.get_feature_names()])
test_df.head()
result=forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output
output.to_csv("submission.csv", index = False, quoting = 3)