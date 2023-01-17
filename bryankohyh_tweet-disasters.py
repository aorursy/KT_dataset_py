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
# data preprocessing for training set



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.head()
from sklearn import feature_extraction, model_selection, linear_model, preprocessing
cv = feature_extraction.text.CountVectorizer()
train_vectors = cv.fit_transform(train["text"])

test_vectors = cv.transform(test["text"])
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()
model = nb.fit(train_vectors.toarray() , train["target"])
model.score(train_vectors.toarray() , train["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = model.predict(test_vectors.toarray())
sample_submission.to_csv("submission.csv", index=False)