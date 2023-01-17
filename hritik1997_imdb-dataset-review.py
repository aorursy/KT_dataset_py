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
dataset1=pd.read_csv("../input/IMDB Dataset.csv") 
dataset1.shape
dataset1.head()
dataset1.shape
dataset=dataset1.sample(frac=1.0)
dataset.shape
dataset.head()
dataset.review.iloc[0]
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus=[]

for i in range(0,50000):

  review1=re.sub('[^a-zA-Z]',' ',dataset.review.iloc[i])

  review1=review1.lower()

  review1=review1.split()

  ps=PorterStemmer()

  review1=[ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]

  review1=' '.join(review1)

  corpus.append(review1)
corpus[0]
#simple bag of word model



#from sklearn.feature_extraction.text import CountVectorizer

#cv=CountVectorizer(max_features=1500)

#x=cv.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.7,min_df=50,ngram_range=(1, 2))

x = vectorizer.fit_transform(corpus)
x.shape
y=dataset.iloc[:,1].values
y.shape
y[999]
for i in range(50000):

    if y[i] == "negative":

        y[i]=1

    else:

        y[i]=0

        

        

    
y=y.astype('int')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
x_train.shape
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(solver='lbfgs')

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm
(6802+6650)/(15000)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm
(6645+6399)/(15000)