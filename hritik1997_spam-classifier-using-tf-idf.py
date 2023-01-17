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
dataset1=pd.read_csv("../input/spam-filter/emails.csv") 
dataset1.head()
dataset1.text.iloc[0]
dataset1.shape
import re

import nltk



nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer


corpus=[]

for i in range(0,5728):

  review1=re.sub('[^a-zA-Z]',' ',dataset1.text.iloc[i])

  review1=review1.lower()

  review1=review1.split()

  ps=PorterStemmer()

  review1=[ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]

  review1=' '.join(review1)

  corpus.append(review1)
corpus[0]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_df=0.7,min_df=10,ngram_range=(1, 2))

x = vectorizer.fit_transform(corpus)
x.shape
y=dataset1.iloc[:,1].values
y.shape
y[0]
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
(1298+384)/(1298+384+5+32)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm
(1302+374)/(42+374+1+1302)