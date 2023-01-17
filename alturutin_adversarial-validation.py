import sys

sys.path.append("../input/moa-scripts")

from moa import load_datasets, preprocess, split

from metrics import logloss

import lgbm as lgb_tools

from lightgbm import LGBMClassifier



import numpy as np

import pandas as pd

import warnings

import joblib

import gc

import matplotlib.pyplot as plt 

import seaborn as sns

from tqdm.auto import tqdm



sns.set_style("dark", {"axes.facecolor": ".92"})

%matplotlib inline

warnings.filterwarnings("ignore")
X, y, _, test, _ = load_datasets("../input/lish-moa")

X, _, test, test_control = preprocess(X, y, test, standard=False, onehot=True)

test = test[~test_control]

test = pd.DataFrame(test, columns=X.columns)

n_train = len(X)

X = pd.concat([X, test])

X['target'] = 0

X['target'].iloc[n_train:] = 1

X.reset_index(drop=True, inplace=True)

del test; gc.collect()



X.shape, X.target.sum(), X.shape[0] - X.target.sum()
class AdversarialValidation:

    """

    Simple class for adversarial analysis

    Bucket and compare each feature from train and test samples using PSI or KL

    PSI: https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf

    KL : https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

    """

    def __init__(self, eps=1-3, target='target'):

        self.target = target

        self.eps = eps

    

    def breakpoints(self, x, n_bins, btype):

        if btype == 'bins':

            min, max = x.min(), x.max()

            return np.linspace(min, max, n_bins)

        elif btype == 'quantiles':

            qnt = np.linspace(0, 100, n_bins)

            return np.stack([np.percentile(x, q) for q in qnt])

        return btype



    def kullback_leibler(self, p, q) -> float:

        lg = np.log(p / q)

        lg = np.where(np.isinf(lg), 0, lg)

        lg = np.where(np.isnan(lg), 0, lg)

        return np.sum(p * lg)



    def psi(self, p, q) -> float:

        lg = np.log(p / q)

        lg = np.where(np.isinf(lg), 0, lg)

        lg = np.where(np.isnan(lg), 0, lg)

        return np.sum((p - q) * lg)

    

    def difference(self, X, method='kl', buckets=10, buckettype='bins'):

        result = {}

        target_mask = X[self.target] == 1

        Na = target_mask.sum() #  `actual`: take test sample as real data

        Ne = len(X) - Na       # `expected`: take train sample as standard

        for c in tqdm(X.columns, desc=method):

            if c == self.target: 

                continue

            a, e = X.loc[target_mask, c].values, X.loc[~target_mask, c].values

            breakpoints = self.breakpoints(X[c].copy(), buckets, buckettype)

            e = np.histogram(e, breakpoints)[0] / Ne

            a = np.histogram(a, breakpoints)[0] / Na

            r = self.kullback_leibler(a, e) if method=='kl' else self.psi(a, e)

            result[c] = r

        return result

av = AdversarialValidation()

diff = av.difference(X, 'psi', 10, 'quantiles')
features = list(diff)

susp = list(zip(*sorted(diff.items(), key=lambda x: x[1])[-16:][::-1]))[0]



from pprint import pprint

pprint(sorted(diff.items(), key=lambda x: x[1])[-16:][::-1])
test_mask = X.target == 1



fig, axes = plt.subplots(4, 4, figsize=(20, 20))

axes=axes.flatten()

for i in range(4*4):

    sns.distplot(X.loc[~test_mask, susp[i]], label='train', hist=0, ax=axes[i])

    sns.distplot(X.loc[test_mask, susp[i]], label='test', hist=0, ax=axes[i])

    axes[i].legend()

sns.despine()
np.random.seed(1)



X = X.sample(frac=1.0, random_state=1)

y = X['target'].values

X.drop('target', axis=1, inplace=True)

X_train, y_train = X.iloc[:15000], y[:15000]

X_valid, y_valid = X.iloc[15000:], y[15000:]
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(C=0.01, random_state=1)

lr.fit(X_train, y_train)

print(f"Adversarial AUC [LR]:{roc_auc_score(y_valid, lr.predict_proba(X_valid)[:, 1])}")



gbm = LGBMClassifier(num_leaves=7, seed=1)

gbm.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=5, verbose=0);

from sklearn.metrics import roc_auc_score

print(f"Adversarial AUC[GBM]:{roc_auc_score(y_valid, gbm.predict_proba(X_valid)[:, 1])}")
gbm_susp = [f for i, f in zip(gbm.feature_importances_, features) if i]

set(susp) & set(gbm_susp) # psi and gbm feature sets not intersect (almost)
def eval_selection(selection, label):

    xt, xv = X_train.copy().drop(selection, axis=1), X_valid.copy().drop(selection, axis=1)

    

    lr = LogisticRegression(C=0.01, random_state=1)

    lr.fit(xt, y_train)

    print(f"Adversarial AUC [LR]-[{label}]:{roc_auc_score(y_valid, lr.predict_proba(xv)[:, 1])}")    

    

    gbm = LGBMClassifier(num_leaves=7, seed=1)

    gbm.fit(xt, y_train, eval_set=(xv, y_valid), early_stopping_rounds=5, verbose=0);

    print(f"Adversarial AUC[GBM]-[{label}]:{roc_auc_score(y_valid, gbm.predict_proba(xv)[:, 1])}")
eval_selection(list(susp), 'PSI')

eval_selection(gbm_susp, 'GBM')

eval_selection(list(set(susp)|set(gbm_susp)), 'PSI+GBM')
# PSI features

susp
# GBM features

gbm_susp