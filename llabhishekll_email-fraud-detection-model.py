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
# importing Natural Language Toolkit 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
df = pd.read_csv('../input/fraud_email_.csv')
df.head()
df.isnull().any()
df = df.dropna()
import nltk
nltk.download('stopwords')
stopset = set(stopwords.words("english"))
vectorizer = TfidfVectorizer(stop_words=stopset,binary=True)
# Extract feature column 'Text'
X = vectorizer.fit_transform(df.Text)
# Extract target column 'Class'
y = df.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)
clf = RandomForestClassifier(n_estimators=15)
y_pred = clf.fit(X_train, y_train).predict_proba(X_test)
print(average_precision_score(y_test ,y_pred[:, 1]))