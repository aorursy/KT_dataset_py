import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('bmh')

plt.rcParams['figure.figsize'] = (10, 10)

title_config = {'fontsize': 20, 'y': 1.05}
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
X_train = train.iloc[:, 2:].values.astype('float64')

y_train = train['target'].values

X_test = test.iloc[:, 1:].values.astype('float64')
from sklearn.base import BaseEstimator, ClassifierMixin

from scipy.special import logsumexp



class RadiusNeighborsNB(BaseEstimator, ClassifierMixin):

    def __init__(self, radius=1.0, steps=100, threshold=500):

        self.radius = radius

        self.steps = steps

        self.threshold = threshold

    def fit(self, X, y):

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        self.log_prior_ = np.log(np.bincount(y)) - np.log(len(y))

        self.grid_ = np.linspace(-5, 5, self.steps)

        # shape of self.log_prob_grid_

        shape = (self.steps, X.shape[1], len(self.log_prior_))

        self.log_prob_grid_ = np.full(shape, self.log_prior_)

        for i in range(shape[0]):

            for j in range(shape[1]):

                mask = np.abs(X[:, j] - self.grid_[i]) < self.radius

                total = mask.sum()

                if total >= self.threshold:

                    self.log_prob_grid_[i, j] = (np.log(np.bincount(y[mask]))

                                                 - np.log(total))

        return self

    def predict_proba(self, X):

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # shape of log_prob

        shape = (*X.shape, len(self.log_prior_))

        log_prob = np.empty(shape)

        for j in range(shape[1]):

            lookup = np.searchsorted(self.grid_, X[:, j])

            lookup[lookup == len(self.grid_)] -= 1

            log_prob[:, j] = self.log_prob_grid_[lookup, j]

        log_posterior = log_prob.sum(axis=1) - (X.shape[1] - 1) * self.log_prior_

        return np.exp(log_posterior - logsumexp(log_posterior))

    def predict(self, X):

        return self.predict_proba(X).argmax(axis=1)
from sklearn.model_selection import StratifiedShuffleSplit



i_train, i_valid = next(StratifiedShuffleSplit(n_splits=1).split(X_train, y_train))
from sklearn.metrics import roc_auc_score



model = RadiusNeighborsNB(radius=0.35, threshold=30)

model.fit(X_train[i_train], y_train[i_train])

print(f'Training AUC is {roc_auc_score(y_train[i_train], model.predict_proba(X_train[i_train])[:, 1])}.')

print(f'Validation AUC is {roc_auc_score(y_train[i_valid], model.predict_proba(X_train[i_valid])[:, 1])}.')
model.fit(X_train, y_train)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = model.predict_proba(X_test)[:, 1]

submission.to_csv('submission.csv', index=False)