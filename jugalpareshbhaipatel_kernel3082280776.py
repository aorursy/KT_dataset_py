# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
tfidf_v = TfidfVectorizer()

train_data = tfidf_v.fit_transform(train_df["text"])



test_data = tfidf_v.transform(test_df["text"])



clf = linear_model.RidgeClassifier()



scores = model_selection.cross_val_score(clf, train_data, train_df["target"], cv=3, scoring="f1")

scores

clf.fit(train_data, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_data)

sample_submission.head()

sample_submission.to_csv("submission.csv", index=False)