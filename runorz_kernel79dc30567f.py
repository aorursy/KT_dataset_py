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
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
from sklearn import svm

clf = svm.SVC()

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
import xgboost as xgb
dtrain = xgb.DMatrix(train_vectors[:7000], label=train_df[:7000]['target'])

dval = xgb.DMatrix(train_vectors[7000:], label=train_df[7000:]['target'])
# res = list()

# for max_depth in range(2,50):

#     params = {'max_depth':max_depth}

#     bst = xgb.train(dtrain = dtrain,params=params)

#     p = bst.predict(dval)

#     p = np.array([1 if e > 0.5 else 0 for e in p])

#     res.append(np.count_nonzero(p == np.array(train_df[7000:]['target'])) / p.shape[0])
import matplotlib.pyplot as plt

plt.plot(res)
dtrain = xgb.DMatrix(train_vectors, label=train_df['target'])

params = {'max_depth':18}

dtest = xgb.DMatrix(test_vectors)

bst = xgb.train(dtrain = dtrain,params=params)

p = bst.predict(dtest)

p = np.array([1 if e > 0.5 else 0 for e in p])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = p

sample_submission.head()

sample_submission.to_csv("submission.csv", index=False)