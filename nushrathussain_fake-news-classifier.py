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
data=pd.read_csv('/kaggle/input/well-shuffled-news-data/Fake_True_appended_and_shuffled.csv')
import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
sw=set(stopwords.words('english'))

stemmed_title=[]

for i in range(0,len(data['title'])):

    temp=(re.sub('[^a-zA-z]',' ',data['title'][i]).lower())

    temp=temp.split()

    ps=PorterStemmer()

    temp=[ps.stem(word) for word in temp if not word in sw]

    temp=' '.join(temp)

    stemmed_title.append(temp)
stemmed_title[0] #glance at stemmed text
# Vectorizing the stemmed_titles----



cv=CountVectorizer()

features=cv.fit_transform(stemmed_title).toarray()

label=data['status']
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB

x=cross_val_score(MultinomialNB(),features,label,cv=5)

print(x)

print('mean : ',x.mean())
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report



x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.1,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.2,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.25,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.33,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
tv=TfidfVectorizer()

features=tv.fit_transform(stemmed_title).toarray()

label=data['status']

from sklearn.model_selection import cross_val_score

x=cross_val_score(MultinomialNB(),features,label,cv=5)

print(x)

print('mean : ',x.mean())
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report



x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.1,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report



x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.2,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report



x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.25,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report



x_train,x_test,y_train,y_test=train_test_split(features,label,test_size=0.33,random_state=0)

classifier=MultinomialNB()

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))