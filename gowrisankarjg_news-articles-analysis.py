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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
df=pd.read_csv("/kaggle/input/news-articles/Articles.csv",encoding="ISO-8859-1")
df.head()
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s
df = df.sample(frac=1).reset_index(drop=True)
df["Article"] = df["Article"].str.replace("strong>","")
df['Article'] = df['Article'].apply(cleaning)
X = df['Article']
encoder = LabelEncoder()
y = encoder.fit_transform(df['NewsType'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
#  K Nearest Neighbours
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_score=knn.score(X_test, y_test)
knn_score
# Multi-layer Perceptron Classifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(6,), random_state=1)
clf.fit(X_train,y_train)
clf_score=clf.score(X_test, y_test)
clf_score
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr_score=lr.score(X_test, y_test)
lr_score
# Naive Bayes
gnb = MultinomialNB()
gnb.fit(X_train, y_train)
gnb_score=gnb.score(X_test, y_test)
gnb_score
d={'Model':[" K-Nearest Neighbours","Multi-layer Perceptron Classifier","Logistic Regression","Naive Bayes"],"Score":[knn_score,clf_score,lr_score,gnb_score]}
final=pd.DataFrame(d)
final