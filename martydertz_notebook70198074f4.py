# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
tweets = pd.read_csv("../input/tweets.csv")

tweets.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df = .8, min_df = .05)
terms = tfidf_vect.fit_transform(tweets.text).toarray()

vocab = tfidf_vect.get_feature_names()

vocab[:10]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_df = .95,min_df = .025)

hrc_counts = vectorizer.fit_transform(hrc.text).toarray()

dt_counts = np.array(vectorizer.fit_transform(dt.text))

vocab = np.array(vectorizer.get_feature_names())

print(hrc_counts.shape, vocab.shape)