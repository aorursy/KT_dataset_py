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
dataset1=pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv") 
dataset1.shape
dataset1.head()
dataset=dataset1.sample(frac=1.0)
dataset.shape
dataset.head()
dataset.review.iloc[1]
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
stopWords=set(stopwords.words('english'))
for i in range(0,50000):
    review1=re.sub('[^a-zA-Z]',' ',dataset.review.iloc[i])
    review1=review1.lower()
    review1=review1.split()
    ps=PorterStemmer()
    review1=[ps.stem(word) for word in review1 if not word in stopWords ]
    review1=' '.join(review1)
    corpus.append(review1)
corpus[1]
#simple bag of word model

#from sklearn.feature_extraction.text import CountVectorizer
#cv=CountVectorizer(max_features=1500)
#x=cv.fit_transform(corpus).toarray()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.8,min_df=50,ngram_range=(1, 2))
x = vectorizer.fit_transform(corpus)
x.shape
y=dataset.iloc[:,1].values
y.shape
y[9999]
for i in range (100):
    print (y[i])
for i in range(50000):
    if y[i] == "negative":
        y[i]=1
    else:
        y[i]=0
y=y.astype('int')
for i in range (100):
    print (y[i])
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)
xtrain.shape
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(solver='lbfgs')
classifier.fit(xtrain,ytrain)
y_pred=classifier.predict(xtest)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
cm
accuracy=(5630+5559)/(5630+581+730+5559)
accuracy
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(xtrain, ytrain)
y_pred=clf.predict(xtest)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_pred)
cm
accur=(5522+5306)/(5522+689+983+5306)
accur
