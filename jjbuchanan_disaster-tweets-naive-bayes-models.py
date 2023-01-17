# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB



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

y = train.target



X_test = test.text
pipe_multinomial = Pipeline([('vectorizer', TfidfVectorizer()),

                 ('clf', MultinomialNB())])



pipe_bernoulli = Pipeline([('vectorizer', CountVectorizer(binary=True)),

                 ('clf', BernoulliNB(binarize=None))])
param_grid_bernoulli = dict(clf__alpha=[0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0])

grid_search_bernoulli = GridSearchCV(pipe_bernoulli,

                                       param_grid=param_grid_bernoulli,

                                       scoring='f1', verbose=1, n_jobs=-1, cv=5)

grid_search_bernoulli.fit(X, y)
cv_results = pd.DataFrame(grid_search_bernoulli.cv_results_)

cv_results.sort_values('rank_test_score').head()
clf_bernoulli = Pipeline([('vectorizer', CountVectorizer(binary=True)),

                 ('clf', BernoulliNB(alpha=0.325, binarize=None))])



f1_scores = cross_val_score(clf_bernoulli, X, y, scoring='f1', cv=5)

mean_f1 = f1_scores.mean()

mean_f1_err = f1_scores.std() / np.sqrt(len(f1_scores))

print(f'f1 score (5-fold CV): {mean_f1:.3f} +/- {mean_f1_err:.3f}')
clf_bernoulli.fit(X, y)

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = clf_bernoulli.predict(X_test)

submission.head()
submission.to_csv("submission_bernoulliNB.csv", index=False)
param_grid_multinomial = dict(vectorizer__binary=[True, False],

    vectorizer__use_idf=[True, False],

    vectorizer__norm=['l2','l1',False],

    vectorizer__smooth_idf=[True,False],

    vectorizer__sublinear_tf=[True,False],

    clf=[MultinomialNB(), ComplementNB()],

    clf__alpha=[0.03, 0.1])

grid_search_multinomial = GridSearchCV(pipe_multinomial,

                                       param_grid=param_grid_multinomial,

                                       scoring='f1', verbose=1, n_jobs=-1, cv=5)

grid_search_multinomial.fit(X, y)
cv_results = pd.DataFrame(grid_search_multinomial.cv_results_)

cv_results.sort_values('rank_test_score').head(10)
clf_multinomial = Pipeline([('vectorizer', TfidfVectorizer(norm='l1', use_idf=False)),

                 ('clf', ComplementNB(alpha=0.029))])



f1_scores = cross_val_score(clf_multinomial, X, y, scoring='f1', cv=5)

mean_f1 = f1_scores.mean()

mean_f1_err = f1_scores.std() / np.sqrt(len(f1_scores))

print(f'f1 score (5-fold CV): {mean_f1:.3f} +/- {mean_f1_err:.3f}')
clf_multinomial.fit(X, y)

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = clf_multinomial.predict(X_test)

submission.head()
submission.to_csv("submission_multinomialNB.csv", index=False)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier



# Best model from linear models notebook

clf_logistic = Pipeline([('countvectorizer', CountVectorizer(binary=True)),

                 ('tfidf', TfidfTransformer()),

                 ('clf', SGDClassifier(loss='log'))])
from sklearn.ensemble import VotingClassifier



clf_voting = VotingClassifier(estimators=[('ber', clf_bernoulli),

                                         ('mul', clf_multinomial),

                                         ('log', clf_logistic)],

                             voting='hard')

# Probabilistic estimates from Naive Bayes models are notoriously unreliable,

# so just use hard voting.
f1_scores = cross_val_score(clf_voting, X, y, scoring='f1', cv=5)

mean_f1 = f1_scores.mean()

mean_f1_err = f1_scores.std() / np.sqrt(len(f1_scores))

print(f'f1 score (5-fold CV): {mean_f1:.3f} +/- {mean_f1_err:.3f}')
clf_voting.fit(X, y)

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = clf_voting.predict(X_test)

submission.head()
submission.to_csv("submission_voting.csv", index=False)
from sklearn.linear_model import LogisticRegression

from mlxtend.classifier import StackingCVClassifier



classifiers_for_stacking = [clf_bernoulli,

                           clf_multinomial,

                           clf_logistic]



clf_stacking = StackingCVClassifier(classifiers=classifiers_for_stacking, meta_classifier=LogisticRegression())
f1_scores = cross_val_score(clf_stacking, X, y, scoring='f1', cv=5)

mean_f1 = f1_scores.mean()

mean_f1_err = f1_scores.std() / np.sqrt(len(f1_scores))

print(f'f1 score (5-fold CV): {mean_f1:.3f} +/- {mean_f1_err:.3f}')
clf_stacking.fit(X, y)

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

submission["target"] = clf_stacking.predict(X_test)

submission.head()
submission.to_csv("submission_stacking.csv", index=False)