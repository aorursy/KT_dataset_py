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
df = pd.read_csv('../input/fake-news/train.csv')
df.head()
df.isnull().sum()
df.dropna(axis=0, inplace=True)
df.shape
df.reset_index(inplace=True)
X = df.drop('label', axis=1)
y = df['label']
message = X.copy()
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
corpus = []
for i in range(0, len(message)): 
    review = re.sub('[^a-zA-Z0-9]', ' ', message.text[i])
    review = review.lower()
    review = review.split()
    lem = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [lem.lemmatize(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
corpus[:10]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=7500)
X_new = cv.fit_transform(corpus).toarray()
X_new
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25, random_state=2)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('ROC_AUC_Score:', metrics.roc_auc_score(y_test, y_pred))
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('ROC_AUC_Score:', metrics.roc_auc_score(y_test, y_pred))
print('Training Score:', clf.score(X_train, y_train))
print('Test Score:', clf.score(X_test, y_test))
from sklearn.linear_model import PassiveAggressiveClassifier
clf_1 = PassiveAggressiveClassifier()
clf_1.fit(X_train, y_train)
y_pred = clf_1.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))

print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('ROC_AUC_Score:', metrics.roc_auc_score(y_test, y_pred))

print('Training Score:', clf_1.score(X_train, y_train))
print('Test Score:', clf_1.score(X_test, y_test))