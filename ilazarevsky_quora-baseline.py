# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import naive_bayes, metrics, preprocessing, model_selection

from sklearn import feature_extraction, feature_selection

from sklearn import linear_model

import matplotlib.pyplot as plt

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
train_data.sample(30)
train_data.shape
train_df, test_df = model_selection.train_test_split(train_data, test_size=0.1, stratify=train_data['target'])
train_df.sample(10)
test_df.sample(10)
count_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=50000, stop_words='english')

train_vectors = count_vectorizer.fit_transform(train_df.question_text)

test_vectors = count_vectorizer.transform(test_df.question_text)
feature_names = count_vectorizer.get_feature_names()
y_train, y_test = train_df['target'], test_df['target']
lreg = linear_model.LogisticRegression(penalty='l2',solver='lbfgs', verbose=1, n_jobs=3, class_weight='balanced')
lreg.fit(train_vectors, y_train)
lr_pred = lreg.predict(test_vectors)
print(metrics.classification_report(y_test, lr_pred))
print(lreg.coef_)
weights = lreg.coef_[0]
weights_ordering = weights.argsort()
print('-')

print(*[feature_names[i] for i in weights_ordering[:100]])
print('+')

print(*[feature_names[i] for i in weights_ordering[::-1][:100]])