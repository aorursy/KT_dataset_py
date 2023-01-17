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
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
tweets = train_df[['text', 'target']]

tweets.head() #check
tweets.shape

tweets.target.value_counts()


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus_train = []

for i in range(0, len(train_df)):

    tweets = re.sub('[^a-zA-Z]', ' ', train_df['text'][i])

    tweets = tweets.lower()

    tweets = tweets.split()

    ps = PorterStemmer()

    tweets = [ps.stem(word) for word in tweets if not word in set(stopwords.words('english'))]

    tweets = ' '.join(tweets)

    corpus_train.append(tweets)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

cv = feature_extraction.text.TfidfVectorizer()

example_train_vectors = cv.fit_transform(corpus_train)
print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = cv.fit_transform(corpus_train)

test_vectors = cv.transform(test_df["text"])
#from sklearn.linear_model import LogisticRegression

#clf = LogisticRegression(random_state = 0)

#from sklearn.ensemble import RandomForestClassifier

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)