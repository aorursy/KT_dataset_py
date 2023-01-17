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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
true = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true['status'] = 0
fake['status'] = 1

df = pd.concat([true, fake])
df = df.sample(frac = 0.5)
df
df.title = df.title.str.lower()
df.text = df.text.str.lower()
df = df.drop(columns = ['subject','date'])
df.text = df.title + ' ' + df.text
df = df.drop(columns = ['title'])
df.head()
train, test = train_test_split(df, test_size = 0.3, random_state = 7)
train
test
cv = CountVectorizer(stop_words = 'english')
fitting = list(train.text)
cv.fit(fitting)
features = cv.transform(fitting).toarray()
inv_vocab = {v: k for k, v in cv.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
new_train = pd.DataFrame(features, columns = vocabulary)
new_train
#Finding the least 80000 used words to be removed from the training dataset
to_remove = list(new_train.sum(axis = 0).sort_values()[:65000].index)
new_train = new_train.drop(columns = to_remove)
new_train
svc = LinearSVC()
svc.fit(new_train, train.status)
test_features = cv.transform(list(test.text)).toarray()
new_test = pd.DataFrame(test_features, columns = vocabulary)
new_test = new_test.drop(columns = to_remove)
new_test
ans = svc.predict(new_test)
accuracy_score(ans, test.status)