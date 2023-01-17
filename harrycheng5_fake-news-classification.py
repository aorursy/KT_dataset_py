# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/real-and-fake-news-dataset/news.csv')



print(df.shape)

print(df.head())
X = df['text']

y = df['label']

print('Num of FAKE:', y[y=='FAKE'].shape[0])

print('Num of REAL:', y[y=='REAL'].shape[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.6)



tfidf_train = tfidf.fit_transform(X_train)

tfidf_test = tfidf.transform(X_test)
pac = PassiveAggressiveClassifier(max_iter=50)

pac.fit(tfidf_train, y_train)



y_pred = pac.predict(tfidf_test)



print('Accuracy:', accuracy_score(y_test, y_pred))
print('Matrix:', confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))