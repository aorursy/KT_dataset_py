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
data = pd.read_csv("../input/spam.csv" ,encoding='latin1')
#let's seperate the output and documents
y = data["v1"].values
x = data["v2"].values
from nltk.corpus import stopwords # for excluding the stopwords
import re # for excluding the integers and punctuation
from nltk.stem import PorterStemmer # for finding the root words
ps = PorterStemmer()
ps.stem("joking")  #how port stemmer works
stopword = set(stopwords.words('english')) # list of stopwords
x = [re.sub('[^a-zA-Z]',' ',doc) for doc in x ] #  include only characters and replace other characters with space
 
document = [doc.split() for doc in x ] # split into words
def convert(words) :
  
    current_words = list()
    for i in words :
        if i.lower() not in stopword :
            
            updated_word = ps.stem(i)
            current_words.append(updated_word.lower())
    return current_words
            
document = [ convert(doc)   for doc in document ] # update the documetns
document = [ " ".join(doc) for doc in document] # again join the words into sentences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
xtrain , xtest , ytrain , ytest = train_test_split(document,y)
cv = CountVectorizer(max_features = 1000) # 1000 features we will use
a = cv.fit_transform(xtrain) # fit using training data and transform into matrix form
b= cv.transform(xtest) #transform testing data into matrix form
a.todense()
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(a,ytrain)
clf.score(b,ytest)