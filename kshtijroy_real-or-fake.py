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
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn import model_selection, feature_extraction, linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.snowball import SnowballStemmer

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test= pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train.shape
test.shape
train.head(10)
test.head(10)
train.isnull().sum()
test.isnull().sum()
train.target.value_counts()
train.target.value_counts().plot(kind='bar')
plt.xlabel('Real disaster or not')
plt.ylabel('Count')
plt.title('Count of tweets')
plt.show()
train.keyword.nunique()
plt.figure(figsize=(12,8))
train.keyword.value_counts()[:20].plot(kind='bar')
plt.xlabel("Keyword")
plt.ylabel("Count ")
plt.title("Count vs Keyword",fontsize =20, weight = 'bold')
train.location.value_counts()
plt.figure(figsize=(12,8))
train.location.value_counts()[:20].plot(kind='bar')
plt.xlabel("Location")
plt.ylabel("Count ")
plt.title("Count vs Location",fontsize =20, weight = 'bold')
def lowercase(text):
    text=text.lower()
    return text

train.text=train.text.apply(lowercase)
test.text=test.text.apply(lowercase)

train['text'].head()
def remove_noise(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
train['text'] = train['text'].apply(remove_noise)
test['text'] = test['text'].apply(remove_noise)
from nltk.stem import PorterStemmer
ps= PorterStemmer()
def stemming(text):
    text=text.split()
    stop= stopwords.words('english')
    text= [ps.stem(word) for word in text if word not in set(stop)]
    text= ' '.join(text)
    return text

train['text']= train['text'].apply(stemming)
test['text']= test['text'].apply(stemming)
train.text.head()
count_vectorizer = CountVectorizer()
count_vectorizer.fit(train['text'])

train_vectors = count_vectorizer.fit_transform(train['text'])
test_vectors = count_vectorizer.transform(test['text'])


y= train['target']
model = MultinomialNB()
scores = model_selection.cross_val_score(model, train_vectors, y, cv=3, scoring="f1")
scores
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
y_test=rfr.fit(train_vectors, y)
sub= pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sub['target'] = rfr.predict(test_vectors)
sub.head()
sub.to_csv("submission.csv", index=False)