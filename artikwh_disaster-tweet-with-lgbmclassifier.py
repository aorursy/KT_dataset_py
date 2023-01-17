# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_union



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv ("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv ("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv ("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_df.head()
train_text= train_df["text"]

test_text=  test_df["text"]

all_text = pd.concat([train_text, test_text])
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    ngram_range=(1, 1),

    max_features=30000)

char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    ngram_range=(1, 4),

    max_features=30000)

vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)
vectorizer.fit(all_text)

train_features = vectorizer.transform(train_text)

test_features = vectorizer.transform(test_text)
clf = LGBMClassifier()
#scores = np.mean(cross_val_score(clf, train_features, train_df["target"], cv=3, scoring="f1"))

#scores
clf.fit(train_features, train_df["target"])
pred = clf.predict(test_features)
submission["target"] = pred
submission.to_csv("submission.csv", index= False)