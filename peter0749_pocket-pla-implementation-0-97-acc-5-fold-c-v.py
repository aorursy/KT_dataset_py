# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.head() # 觀察資料
# malignant: +1
# benign: -1
X = np.array(data.iloc[:,2:-1], dtype=np.float32)
Y = np.array(data.iloc[:,1]=='M', dtype=np.int16)*2-1 # +1: malignant; -1: benign
print(X.shape) # Check feature shape
print(Y.shape) # Check label shape
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

kfold = StratifiedKFold(n_splits=5, shuffle=True) # 5-fold cross validation
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
pca   = PCA(n_components=15) # sum of explained_variance_ratio_ > 0.97
class PLA(object):
    def __init__(self, x_dim, eta=1.0, shuffle=False, verbose=False):
        self.shuffle = shuffle
        self.verbose = verbose
        self.eta = eta
        self.Wxb = np.random.normal(0, np.sqrt(2/(x_dim+1)), size=(1,x_dim+1)) # initialize Wxb using he-normal
    def predict(self, x, pocket=False):
        X = np.append(x, [1], axis=-1)[...,np.newaxis]
        pred = np.squeeze(self.Wxb @ X)
        return -1 if pred<=0 else 1
    def train(self, Xs, Ys):
        updates = 0
        correct_cnt = 0
        i = 0
        while correct_cnt<len(Xs): # cyclic method
            if self.shuffle and correct_cnt==0:
                idx = np.random.permutation(len(Xs))
                Xs, Ys = Xs[idx], Ys[idx] # faster
                i = 0
            x, y = Xs[i], Ys[i]
            p = self.predict(x)
            if p!=y: # wrong
                self.Wxb = self.Wxb + (self.eta*y*np.append(x, [1], axis=-1))[np.newaxis]
                updates += 1
                if self.verbose:
                    print('iteration {:d}: '.format(updates), self.Wxb)
                correct_cnt = 0
            else:
                correct_cnt += 1
            i = (i+1)%len(Xs)
        return updates

class PocketPLA(PLA):
    def __init__(self, x_dim, eta=1.0, pocket_maxiter=None, shuffle=False, verbose=False):
        super(PocketPLA, self).__init__(x_dim, eta=eta, shuffle=shuffle, verbose=verbose)
        self.pocket_maxiter = pocket_maxiter
        self.Wxb_pocket = np.zeros_like(self.Wxb, dtype=np.float32) # (1, 4)
    def predict(self, x, pocket=False):
        W = self.Wxb_pocket if pocket else self.Wxb
        X = np.append(x, [1], axis=-1)[...,np.newaxis]
        pred = np.squeeze(W @ X)
        return -1 if pred<=0 else 1
    def train(self, Xs, Ys):
        updates = 0
        last_errors = np.inf
        while True:
            if self.shuffle: # precomputed random order; else: naive cyclic
                idx = np.random.permutation(len(Xs))
                Xs, Ys = Xs[idx], Ys[idx] # faster
            for x, y in zip(Xs, Ys):
                p = self.predict(x)
                if p!=y: # wrong
                    self.Wxb = self.Wxb + (self.eta*y*np.append(x, [1], axis=-1))[np.newaxis]
                    updates += 1
                    break
            errors = 0
            for x, y in zip(Xs, Ys):
                p = self.predict(x)
                errors += 1 if p!=y else 0
            if errors < last_errors:
                last_errors = errors
                self.Wxb_pocket = self.Wxb.copy()
                if self.verbose:
                    print('iteration {:d}: update pocket weights: err: {:.2f}'.format(updates, errors/len(Xs)))
            if updates>=self.pocket_maxiter or last_errors==0:
                return last_errors

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from IPython.display import display
max_iteration = 3000
eta = 0.5
accs = []
precs = []
recs = []
f1s = []
for train, valid in kfold.split(X, Y):
    print('{:d} samples for training, {:d} samples for validation.'.format(len(train), len(valid)))
    X_train, Y_train = X[train], Y[train]
    X_valid, Y_valid = X[valid], Y[valid]
    X_train = pca.fit_transform(scaler.fit_transform(X_train)) # only fit on training set
    X_valid = pca.transform(scaler.transform(X_valid))
    pocket_pla = PocketPLA(X_train.shape[-1], eta=eta, pocket_maxiter=max_iteration, shuffle=True)
    pocket_pla.train(X_train, Y_train) # apply pla
    preds = np.asarray([pocket_pla.predict(x) for x in X_valid], dtype=np.int16) # prediction
    acc = accuracy_score(Y_valid, preds) # evaluations
    precision = precision_score(Y_valid, preds)
    recall = recall_score(Y_valid, preds)
    f1 = f1_score(Y_valid, preds)
    accs.append(acc)
    precs.append(precision)
    recs.append(recall)
    f1s.append(f1)
    print('acc: {:.2f}, precision: {:.2f}, recall: {:.2f}, f1: {:.2f}'.format(acc,precision,recall, f1))
am, pm, rm, fm = np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s)
ad, pd_, rd, fd = np.std(accs)*2, np.std(precs)*2, np.std(recs)*2, np.std(f1s)*2
print('acc: {:.2f}+/-{:.2f}, precision: {:.2f}+/-{:.2f}, recall: {:.2f}+/-{:.2f}, f1: {:.2f}+/-{:.2f}'.format(am, ad, pm, pd_, rm, rd, fm, fd))