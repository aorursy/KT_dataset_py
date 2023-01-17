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

from sklearn import *
train = pd.read_csv("../input/nlp-getting-started/train.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer()
train_vectors = tfidf_vectorizer.fit_transform(train["text"])
train_vectors.todense().shape
test_vectors = tfidf_vectorizer.transform(test["text"])
test_vectors.todense().shape
tflm = linear_model.RidgeClassifier()
validation_scores = model_selection.cross_val_score(tflm, train_vectors, train["target"], cv=3, scoring="f1")
validation_scores
validation_scores = model_selection.cross_val_score(tflm, train_vectors, train["target"], cv=10, scoring="f1")
validation_scores
sample = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample.head()
tflm.fit(train_vectors,train["target"])
sample["target"] = tflm.predict(test_vectors)
sample.head()
sample.merge(test)
sample
sample.to_csv("submission.csv",index = False)