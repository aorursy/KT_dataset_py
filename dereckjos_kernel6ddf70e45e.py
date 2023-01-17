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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import string
import scipy
import numpy
import nltk
import json
import sys
import csv
import os
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train.head()
train.isnull().sum()
train.drop(columns=["id","keyword","location"],axis=1,inplace=True)
from nltk.corpus import stopwords
stop = stopwords.words('english')
train['text'] = train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

def processRow(row):
 import re
 import nltk
 from textblob import TextBlob
 from nltk.corpus import stopwords
 from nltk.stem import PorterStemmer
 from textblob import Word
 from nltk.util import ngrams
 import re
 from nltk.tokenize import word_tokenize
 tweet = row

#Lower case
 tweet=tweet.lower()

#Removes unicode strings like "\u002c"  -> ,(comma)
 tweet = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', tweet)
    
# Removes non-ascii characters. note : \x00 to \x7f is 00 to 255
# non-ascii characters like copyrigth symbol, trademark symbol
 tweet = re.sub(r'[^\x00-\x7f]',r'',tweet)
               
#convert any url to URL
 tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
               
#Convert any @Username to "AT_USER"
 tweet = re.sub('@[^\s]+','AT_USER',tweet)

#Remove additional white spaces
 tweet = re.sub('[\s]+', ' ', tweet)
 tweet = re.sub('[\n]+', ' ', tweet)

#Remove not alphanumeric symbols white spaces
 tweet = re.sub(r'[^\w]', ' ', tweet)

#Removes hastag in front of a word """
 tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

#Replace #word with word
 tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

# #Removes all possible emoticons
 tweet = re.sub(':\)|:\(|:\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', tweet)
#remove numbers -> this is optional
 tweet = ''.join([i for i in tweet if not i.isdigit()])

#remove multiple exclamation -> this is optional
 tweet = re.sub(r"(\!)\1+", ' ', tweet)

#remove multiple question marks -> this is optional
 tweet = re.sub(r"(\?)\1+", ' ', tweet)

#remove multistop -> this is optional
 tweet = re.sub(r"(\.)\1+", ' ', tweet)

#trim
 tweet = tweet.strip('\'"')
     
#lemma
 from textblob import Word
 tweet =" ".join([Word(word).lemmatize() for word in tweet.split()])

#stemmer
#st = PorterStemmer()
#tweet=" ".join([st.stem(word) for word in tweet.split()])
          
 row = tweet
 return row
train["text"]=train["text"].apply(lambda x: processRow(x))
train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train["text"], train["target"], test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
text_clf_lr=Pipeline([("tdfidf",TfidfVectorizer()),("lr",LogisticRegression())]) 
text_clf_nb=Pipeline([("tdfidf",TfidfVectorizer()),("lr",MultinomialNB())]) 
text_clf_svm=Pipeline([("tdfidf",TfidfVectorizer()),("clf",LinearSVC())]) 
text_clf_rfc=Pipeline([("tdfidf",TfidfVectorizer()),("rfc",RandomForestClassifier(criterion="gini",max_depth=15))])
text_clf_dtc=Pipeline([("tdfidf",TfidfVectorizer()),("dtc",DecisionTreeClassifier())])
text_clf_knn=Pipeline([("tdfidf",TfidfVectorizer()),("knn",KNeighborsClassifier())])
text_clf_svm.fit(X_train,y_train)
text_clf_rfc.fit(X_train,y_train)
text_clf_dtc.fit(X_train,y_train)
text_clf_knn.fit(X_train,y_train)
text_clf_lr.fit(X_train,y_train)
text_clf_nb.fit(X_train,y_train)
pred_dtc=text_clf_dtc.predict(X_test)
pred_knn=text_clf_knn.predict(X_test)
pred_rfc=text_clf_rfc.predict(X_test)
pred_svm=text_clf_svm.predict(X_test)
pred_lr=text_clf_lr.predict(X_test)
pred_nb=text_clf_nb.predict(X_test)
from sklearn.metrics import accuracy_score
print("*****************DECISION TREE CLASSIFIER**************************")
print(accuracy_score(y_test,pred_dtc))
print("***************************KNN CLASSIFIER**************************")
print(accuracy_score(y_test,pred_knn))
print("*****************RANDOM FOREST CLASSIFIER**************************")
print(accuracy_score(y_test,pred_rfc))
print("*******************SUPPORT VECTOR MACHINE**************************")
print(accuracy_score(y_test,pred_svm))
print("**********************Logistic Regression**************************")
print(accuracy_score(y_test,pred_lr))
print("***************************Naive Bayes*****************************")
print(accuracy_score(y_test,pred_nb))