# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_file_path = '../input/train.csv'

valid_file_path = '../input/valid.csv'

sample_file_path = '../input/sample_submission.csv'
train_file = pd.read_csv(train_file_path)

valid_file = pd.read_csv(valid_file_path)

sample_file = pd.read_csv(sample_file_path)
train_file.head(5)
count_vect = CountVectorizer()

text = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='log', penalty='l2', n_iter=1000))])

fit = text.fit(train_file.headline, train_file.is_sarcastic)
predicted = fit.predict(valid_file.headline)

sample_file["is_sarcastic"] = predicted

sample_file.to_csv("submission.csv", index = False)

sample_file.describe()