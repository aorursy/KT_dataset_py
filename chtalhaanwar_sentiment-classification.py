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
columns_name = ["target", "ids", "date", "flag", "user", "text"]

df = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv", encoding = 'ISO-8859-1', names = columns_name)

df.head()
user_tweets=list(df['text'])
df['target']=df['target'].map({0:0,4:1})
df['target'].value_counts()
labels=df['target'].values
import re

import string

clean_tweets =[i.lower() for i in user_tweets] 

clean_tweets=[re.sub('RT @\w+:'," ", i) for i in clean_tweets]

clean_tweets=[re.sub('@(\w+)'," ", i) for i in clean_tweets]

clean_tweets =[re.sub('\d'," ", i) for i in clean_tweets] 

clean_tweets =[re.sub('http\S+'," ", i) for i in clean_tweets] 

clean_tweets =[i.translate(str.maketrans('', '', string.punctuation)) for i in clean_tweets] 

clean_tweets =[re.sub('\s+'," ", i) for i in clean_tweets] 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

class_vector = vectorizer.fit_transform(clean_tweets)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(class_vector,labels,test_size=0.3)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=500)

lr.fit(X_train,y_train)



y_pred=lr.predict(X_test)


from sklearn.metrics import classification_report

print(classification_report(y_pred,y_test))



import pickle

with open('class_tfidf_vec.pickle', 'wb') as fin:

    pickle.dump(vectorizer, fin)



import pickle

with open('class_lr_clf.pickle', 'wb') as fin:

    pickle.dump(lr, fin)