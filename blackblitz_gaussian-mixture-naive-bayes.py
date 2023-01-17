import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('bmh')

plt.rcParams['figure.figsize'] = (20, 10)

title_config = {'fontsize': 20, 'y': 1.05}
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
X_train = train.iloc[:, 2:].values.astype('float64')

y_train = train['target'].values

X_test = test.iloc[:, 1:].values.astype('float64')
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler



for i in range(2):

    fig, axes = plt.subplots(3, 6)

    axes = axes.ravel()

    for j in range(len(axes)):

        feature = StandardScaler().fit_transform(X_train[y_train == i, j:j + 1])

        hist = axes[j].hist(feature, bins='auto', histtype='step',

                            linewidth=2, density=True)

        grid = np.linspace(feature.min(), feature.max(), num=1000)

        log_density = (GaussianMixture(n_components=10, reg_covar=0.03)

                       .fit(feature).score_samples(grid[:, None]))

        gmm = axes[j].plot(grid, np.exp(log_density))

        axes[j].set_title(f'var_{j}', **title_config)

        axes[j].set_ylim([0, 1])

    fig.suptitle(f'Histogram vs Gaussian Mixture Model for Class {i}',

                 **title_config)

    fig.legend((hist[2][0], gmm[0]), ('Histogram', 'Gaussian mixture model'),

               loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=14)

    fig.tight_layout()

    fig.subplots_adjust(top=0.88)
from sklearn.base import BaseEstimator, ClassifierMixin

from scipy.special import logsumexp



class GaussianMixtureNB(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components=1, reg_covar=1e-06):

        self.n_components = n_components

        self.reg_covar = reg_covar

    def fit(self, X, y):

        self.log_prior_ = np.log(np.bincount(y) / len(y))

        # shape of self.log_pdf_

        shape = (len(self.log_prior_), X.shape[1])

        self.log_pdf_ = [[GaussianMixture(n_components=self.n_components,

                                          reg_covar=self.reg_covar)

                          .fit(X[y == i, j:j + 1])

                          .score_samples for j in range(shape[1])]

                         for i in range(shape[0])]

    def predict_proba(self, X):

        # shape of log_likelihood before summing

        shape = (len(self.log_prior_), X.shape[1], X.shape[0])

        log_likelihood = np.sum([[self.log_pdf_[i][j](X[:, j:j + 1])

                                  for j in range(shape[1])]

                                 for i in range(shape[0])], axis=1).T

        log_joint = self.log_prior_ + log_likelihood

        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def predict(self, X):

        return self.predict_proba(X).argmax(axis=1)
from sklearn.model_selection import StratifiedShuffleSplit



i_train, i_valid = next(StratifiedShuffleSplit(n_splits=1).split(X_train, y_train))
from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score



pipeline = make_pipeline(StandardScaler(),

                         GaussianMixtureNB(n_components=10, reg_covar=0.03))

pipeline.fit(X_train[i_train], y_train[i_train])

print('Training AUC is {}.'

      .format(roc_auc_score(y_train[i_train],

                            pipeline.predict_proba(X_train[i_train])[:, 1])))

print('Validation AUC is {}.'

      .format(roc_auc_score(y_train[i_valid],

                            pipeline.predict_proba(X_train[i_valid])[:, 1])))
pipeline.fit(X_train, y_train)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = pipeline.predict_proba(X_test)[:, 1]

submission.to_csv('submission.csv', index=False)