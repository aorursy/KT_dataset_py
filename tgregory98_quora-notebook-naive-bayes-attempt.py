# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
path = "../input" # Change to "../input" on Kaggle, or "../../LOCAL FILES/all" on local
print(os.listdir(path))

# Any results you write to the current directory are saved as output.

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import metrics
from sklearn.metrics import f1_score
train = pd.read_csv(path + "/train.csv")
test = pd.read_csv(path + "/test.csv")
sample = pd.read_csv(path + "/sample_submission.csv")
count_vectorizer = CountVectorizer()
count_vectorizer.fit(train['question_text'].append(test['question_text']))
train_data = count_vectorizer.transform(train['question_text'])
test_data = count_vectorizer.transform(test['question_text'])

train_targets = train['target']
kf = KFold(n_splits=4, shuffle = False, random_state = 42)
print(kf)
predictions = np.zeros((train.shape[0], ))
final_predictions = 0

for train_index, test_index in kf.split(train):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_data[train_index], train_data[test_index]
    y_train, y_test = train_targets[train_index], train_targets[test_index]
    model = MultinomialNB()
    model.fit(X_train,y_train)
    predictions[test_index] = model.predict_proba(X_test)[:, 1]
    final_predictions = final_predictions + model.predict_proba(test_data)[:, 1]
for thresh in np.arange(0.1, 0.901, 0.02):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(train_targets, (predictions>thresh).astype(int))))
predictions = (predictions > 0.68).astype(np.int)
final_predictions = (final_predictions > 0.68).astype(np.int)
f1_score(train_targets, predictions)
output = pd.DataFrame({"qid":test["qid"].values})
output['prediction'] = final_predictions
output.to_csv("submission.csv", index=False)