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
real = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
print(real.head())
print(fake.head())

real_title = real['title'].to_numpy()
fake_title = fake['title'].to_numpy()
corpus = np.hstack((real_title, fake_title))
y = np.array([1]*real_title.shape[0] + [0]*fake_title.shape[0])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
X = tfidf.fit_transform(corpus)
X.shape
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

train_X, val_X, train_y, val_y = train_test_split(corpus, y)
count_vect = CountVectorizer()
train_X_counts = count_vect.fit_transform(train_X)
tfidf_transformer = TfidfTransformer()
train_X_tfidf = tfidf_transformer.fit_transform(train_X_counts)
clf = MultinomialNB().fit(train_X_tfidf, train_y)

from sklearn.metrics import mean_absolute_error

res_y = clf.predict(count_vect.transform(val_X))
mean_absolute_error(res_y, val_y)
