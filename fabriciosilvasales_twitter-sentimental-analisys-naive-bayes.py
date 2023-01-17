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
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split
data1 = pd.read_csv('/kaggle/input/twitter-and-reddit-sentimental-analysis-dataset/Twitter_Data.csv')
data1 = data1.dropna()
tweets = list(data1['clean_text']) 
classes = []

for r in data1["category"]:
    if  r == -1:
        classes.append("Negative")
    if  r == 0:
        classes.append("Neutral")
    if  r == 1:
        classes.append("Positive")
    if r != 0 and r!= 1 and r != -1:
          print(r)
    
base = pd.DataFrame({"tweet": tweets, "category": classes})
base
tweets = base["tweet"]
classes = base['category']

def Preprocessing(data):
    stemmer = nltk.stem.RSLPStemmer()
    data = re.sub(r"http\S+", "", data).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [stemmer.stem(i) for i in data.split() if not i in stopwords]
    return (" ".join(words))

tweets = [Preprocessing(i) for i in tweets]

vectorizer = CountVectorizer(analyzer="word")
freq_tweets_train = vectorizer.fit_transform(tweets)
#freq_tweets_test = vectorizer.fit_transform(X_test)
model = MultinomialNB()
model.fit(freq_tweets_train,y_train)
for t, c in zip (X_test,model.predict(freq_tweets_test)):
    print (t +", "+ c)