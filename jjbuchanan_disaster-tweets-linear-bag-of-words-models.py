# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.sample(5)
test.sample(5)
X = train.text

X_test = test.text
y = train.target
P = sum(train.target) / len(train)
print('F1 score for Bernoulli guessing:')

print(f'{P:.3f}')
print('F1 score for constant guessing:')

print(f'{2*P/(P+1):.3f}')
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = 1

submission.head()
submission.to_csv("submission_naive.csv", index=False)
# Construct a bag-of-words representation

countvectorizer = CountVectorizer()

X = countvectorizer.fit_transform(X)
# Utility method

def print_item_from_sparse(sparse_matrix):

    A = sparse_matrix[0].toarray()[0]

    for i in range(len(A)):

      if A[i] != 0:

        print(f'{i}: {A[i]}')
# See what a tweet looks like in this representation

print_item_from_sparse(X)
# tf-idf reweighting

tfidf = TfidfTransformer()

X = tfidf.fit_transform(X)



print_item_from_sparse(X)
clf = SGDClassifier()



f1_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')

mean_f1 = f1_scores.mean()

mean_f1_err = f1_scores.std() / np.sqrt(len(f1_scores))

print(f'f1 score (5-fold CV): {mean_f1:.3f} +/- {mean_f1_err:.3f}')
X = train.text
pipe = Pipeline([('countvectorizer', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('clf', SGDClassifier())])
# Here's a sample parameter grid to look at.

# Notice that we can add or remove entire steps in the Pipeline,

# which we try here with TfidfTransformer.

param_grid = dict(countvectorizer__binary=[True, False],

    tfidf=['passthrough', TfidfTransformer()],

    clf__loss=['log', 'hinge', 'modified_huber', 'squared_loss'],

    clf__penalty=['l1', 'l2'],

    clf__alpha=[0.001, 0.0001, 0.00001])

grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring='f1',

                           verbose=2, n_jobs=-1, cv=5)
grid_search.fit(X, y)
cv_results = pd.DataFrame(grid_search.cv_results_)

cv_results.sort_values('rank_test_score').head()
grid_search.best_params_
best_model = Pipeline([('countvectorizer', CountVectorizer(binary=True)),

                 ('tfidf', TfidfTransformer()),

                 ('clf', SGDClassifier(loss='log'))])
f1_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1')

mean_f1 = f1_scores.mean()

mean_f1_err = f1_scores.std() / np.sqrt(len(f1_scores))

print(f'f1 score (5-fold CV): {mean_f1:.3f} +/- {mean_f1_err:.3f}')
best_model.fit(X, y)
submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = best_model.predict(X_test)

submission.head()
submission.to_csv("submission_linear_bagofwords.csv", index=False)