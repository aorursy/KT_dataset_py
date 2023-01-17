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

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
test.head(100)
train.head()
test.info()

train.info()
train.shape

test.shape
train.isnull().sum()
train = train.drop(['location','keyword'], axis=1)
train.info()
train.isnull().sum()
train = train.drop('id', axis=1)
train.info()
test = test.drop(['id','location','keyword'], axis=1)
test.info()
test.isnull().sum()
train["text"].head(50)


import re

from string import punctuation

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize



stop_words = set(stopwords.words('english')) 



def remove_noise_from_text(review):

    review = str(review)

    review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)

    words = review.split()

    new_review = str()

    for word in words:

        if word in stop_words:

            continue;

        else:

            word = word.lower()

            word = word.strip(punctuation)

            word = word.strip()

            new_review += word + " "

    return new_review[:len(new_review)-1]
train['clean_text'] = train['text'].apply(remove_noise_from_text)

train.head(50)
test['clean_text'] = test['text'].apply(remove_noise_from_text)

test.head(50)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer



x_train = train['clean_text']

target = train['target']

X_test = test['clean_text']



pipeline = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', KNeighborsClassifier(n_neighbors=20)),

])



pipeline.fit(x_train, target)

sample_submission["target"] = pipeline.predict(X_test)

sample_submission.to_csv("submission.csv", index=False)