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
import numpy as np
import pandas as pd
df = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.info()
# checking for missing values
df['review'].isna().sum(), df['sentiment'].isna().sum()
df['sentiment'][:5]
# convert sentiment to numbers

df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

df.head()
# Convert reviews into vectors

sentences = df['review']

import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
type(sentences), sentences[0]
%%time

# tokanize
import re

for index, line in enumerate(sentences):
    if index % 1000 == 0:
        print(f'line: {index}/{len(sentences)}')
    new_line = re.sub('[,"<>/-]', '', line)
    sentences[index] = ' '.join(word.lower() for word in line.split() if word not in stopwords)

# sentences = sentences.map(lambda line: [word.lower() for word in line.replace(',', '').split() if word not in stopwords])
    
sentences[0]
df['reviews'] = sentences
len(df['reviews']), len(df['sentiment'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['reviews'], df['sentiment'], test_size = 0.2)

len(x_train), len(x_test), len(y_train), len(y_test)
x_train, y_train, type(y_train), df.iloc[:0]
%%time
from sklearn.feature_extraction.text import CountVectorizer

vc = CountVectorizer()
x_tr = vc.fit_transform(x_train)
# x_ts = vc.fit_transform(x_test)
# x_tr[0], x_tr[1]
x_tr.shape, len(y_train), x_tr[8].shape

for a in x_tr:
    if a.shape[1] != 92759:
        print(False)
else:
    print(True)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

y_pred = lr.fit(x_tr, y_train).predict(vc.transform(x_test))
print(accuracy_score(y_test, y_pred))
%%time

from sklearn.svm import LinearSVC
svc = LinearSVC(C=0.5,random_state=42,max_iter=10000)

y_pred = svc.fit(x_tr, y_train).predict(vc.transform(x_test))
print(accuracy_score(y_test, y_pred))
%%time
from sklearn.naive_bayes import GaussianNB

gb = GaussianNB()
y_pred = gb.fit(x_tr, y_train).predict(vc.transform(x_test))
print(accuracy_score(y_test, y_pred))
print('done')
