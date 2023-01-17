# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
messages=pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding = "ISO-8859-1")
messages.head()
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['v2'][i])
    review=review.lower()
    review=review.split()
    review=[stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
print(corpus)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=5000)
X=vectorizer.fit_transform(corpus).toarray()
print(X)
y=pd.get_dummies(messages['v1'])
y=y.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression()
LR=LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)
print(accuracy)
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


classifier=classifier.fit(X_train,y_train)
y_pred_NB=classifier.predict(X_test)
accuracy_NB=accuracy_score(y_pred_NB,y_test)
print(accuracy_NB)
