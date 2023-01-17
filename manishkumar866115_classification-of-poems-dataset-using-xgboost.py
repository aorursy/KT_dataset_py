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
file_path="../input/poems.csv"

with open(file_path, 'rb') as f:

  contents = f.read()



contents=contents.decode(encoding='utf-8',errors='ignore')

#print(contents)
from io import StringIO



TESTDATA = StringIO(contents)



dataset = pd.read_csv(TESTDATA, sep=",",error_bad_lines=False)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

dataset["category"]=le.fit_transform(dataset["category"])
print(dataset.describe())
print(dataset.iloc[1])
#creating our own corpus with the help of word tokenizer, porter stemmer,stopwords, regular expressions and nltk

import re   

import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.stem import PorterStemmer

corpus=[]

ps=PorterStemmer()

stopwords=set(stopwords.words("english"))

for i in range(394):

    review=dataset.iloc[i][1]

    review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[i][1] )

    review = re.sub('[?&#@$%!_.|,-:;"]', ' ', dataset.iloc[i][1] )    

    review=review.lower()

    words=word_tokenize(review)

    d=list()

    for w in words:

        if w not in stopwords:

            d.append(ps.stem(w))

    review=" ".join(d)

    corpus.append(review)

    

#demo of how the corpus looks like

print(corpus[5:6])
#creating bag of word using count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=2000)

X=cv.fit_transform(corpus).toarray()

y=dataset.iloc[:,3].values
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.30,random_state=0,shuffle=True)
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_train, Y_train) 



y_pred = gnb.predict(X_test) 
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(Y_test, Y_pred)

cm2 = confusion_matrix(Y_test, y_pred)

#confusion matrix for xgboost in first with accuracy of nearly 90%

print(cm1)

#confusion matrix for naive bayes with low accuracy of nearly 60%

print(cm2)