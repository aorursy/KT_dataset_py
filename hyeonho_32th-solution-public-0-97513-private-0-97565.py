import warnings

warnings.filterwarnings("ignore")



import subprocess

import re

import sys

import glob

import ctypes

import collections

import os

os.environ["OMP_NUM_THREADS"] = "1"

os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["MKL_NUM_THREADS"] = "1"

os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

os.environ["NUMEXPR_NUM_THREADS"] = "1"



import numpy as np, pandas as pd, gc

import random as rn

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, accuracy_score

from sklearn.decomposition import PCA, KernelPCA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC, NuSVC

from sklearn.preprocessing import *

from sklearn.mixture import GaussianMixture

from sklearn.cluster import AgglomerativeClustering

from copy import copy, deepcopy



from scipy.stats import probplot

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm, tqdm_notebook



%matplotlib inline



pd.options.display.max_rows = 10000

pd.options.display.max_columns = 10000

pd.options.display.max_colwidth = 1000



from IPython.display import display



_MKL_ = 'mkl'

_OPENBLAS_ = 'openblas'
class BLAS:

    def __init__(self, cdll, kind):

        if kind not in (_MKL_, _OPENBLAS_):

            raise ValueError(f'kind must be {MKL} or {OPENBLAS}, got {kind} instead.')

        

        self.kind = kind

        self.cdll = cdll

        

        if kind == _MKL_:

            self.get_n_threads = cdll.MKL_Get_Max_Threads

            self.set_n_threads = cdll.MKL_Set_Num_Threads

        else:

            self.get_n_threads = cdll.openblas_get_num_threads

            self.set_n_threads = cdll.openblas_set_num_threads

            



def get_blas(numpy_module):

    LDD = 'ldd'

    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'



    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')

    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, '_multiarray_umath.*so'))[0]

    ldd_result = subprocess.run(

        args=[LDD, MULTIARRAY_PATH], 

        check=True,

        stdout=subprocess.PIPE, 

        universal_newlines=True

    )



    output = ldd_result.stdout



    if _MKL_ in output:

        kind = _MKL_

    elif _OPENBLAS_ in output:

        kind = _OPENBLAS_

    else:

        return



    pattern = LDD_PATTERN.format(kind)

    match = re.search(pattern, output, flags=re.MULTILINE)



    if match:

        lib = ctypes.CDLL(match.groupdict()['path'])

        return BLAS(lib, kind)

    



class single_threaded:

    def __init__(self, numpy_module=None):

        if numpy_module is not None:

            self.blas = get_blas(numpy_module)

        else:

            import numpy

            self.blas = get_blas(numpy)



    def __enter__(self):

        if self.blas is not None:

            self.old_n_threads = self.blas.get_n_threads()

            self.blas.set_n_threads(1)

        else:

            warnings.warn(

                'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'

            )



    def __exit__(self, *args):

        if self.blas is not None:

            self.blas.set_n_threads(self.old_n_threads)

            if self.blas.get_n_threads() != self.old_n_threads:

                message = (

                    f'Failed to reset {self.blas.kind} '

                    f'to {self.old_n_threads} threads (previous value).'

                )

                raise RuntimeError(message)

    

    def __call__(self, func):

        def _func(*args, **kwargs):

            self.__enter__()

            func_result = func(*args, **kwargs)

            self.__exit__()

            return func_result

        return _func
SEED = 42

N_FOLDS = 11

V_THRES = 1.5

MODIFY_THRES = 0.99999

PSEUDO_LABEL_THRES = 0.99



np.random.seed(SEED)

rn.seed(SEED)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
def execute_classifier(n_folds=N_FOLDS, v_thres=V_THRES, start_seed=SEED, seed_range=4, verbose=None, magic_range=512):

    

    # INITIALIZE VARIABLES

    oof = np.zeros(len(train))

    fake_oof = np.zeros(len(train))

    fake_preds = np.zeros(len(test))

    

    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

    

    # BUILD 512 SEPARATE NON-LINEAR MODELS

    for i in tqdm_notebook(range(magic_range), f'{magic_range} models..'):

        

        for seed in range(start_seed, start_seed+seed_range):

            fix_clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=seed, nu=0.7, coef0=0.05)



            # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I

            train2 = train[train['wheezy-copper-turtle-magic'] == i]

            test2 = test[test['wheezy-copper-turtle-magic'] == i]

            idx1 = train2.index

            idx2 = test2.index

            train2.reset_index(drop=True, inplace=True)



            train3, test3 = gmm_fe(train2[cols], test2[cols], v_thres=v_thres, seed=seed)



            # pre_clf

            pre_clf_oof, pre_clf_preds = pre_clf(train3, train2['target'], test3, fix_clf=fix_clf, n_folds=n_folds, seed=seed)

            oof[idx1] += pre_clf_oof



            # outlier replace

            train4 = outlier_replace(pre_clf_oof, np.concatenate([train3, np.array(train2['target']).reshape(-1, 1)], axis=1))

            # pseudo labeling

            train4 = pseudo_labeling(pre_clf_preds, train4, test3, rows_ratio=0.2)



            clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=seed, nu=0.6, coef0=0.05)



            # modeling

            skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

            for (_, val_idx), (train_idx, _) in zip(skf.split(train2, train2['target']), skf.split(train4, train4.iloc[:, -1])):

                clf.fit(train4.iloc[train_idx, :-1], train4.iloc[train_idx, -1])

                fake_oof[idx1[val_idx]] += clf.predict_proba(train3[val_idx])[:, 1]

                fake_preds[idx2] += clf.predict_proba(test3)[:, 1] / skf.n_splits

        

        # scaling

        fake_oof[idx1] = StandardScaler().fit_transform(fake_oof[idx1].reshape(-1,1)).ravel()

        fake_preds[idx2] = StandardScaler().fit_transform(fake_preds[idx2].reshape(-1,1)).ravel()

        

    return oof, fake_oof, fake_preds
def pre_clf(train, y, test, fix_clf=True, n_folds=N_FOLDS, seed=SEED):

    

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))

    

    skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

    for train_index, valid_index in skf.split(train, y):

        fix_clf.fit(train[train_index], y[train_index])

        oof[valid_index] = fix_clf.predict_proba(train[valid_index])[:, 1]

        preds += fix_clf.predict_proba(test)[:, 1] / skf.n_splits

    

    return oof, preds
def outlier_replace(oof, train, modify_threshold=MODIFY_THRES):

    

    # type change

    oof = pd.Series(oof)

    train = pd.DataFrame(train)

    y = train.iloc[:, -1]

    

    # index extract

    oof2 = oof[(oof>modify_threshold) | (oof<1-modify_threshold)]

    oof2[oof2>0.5]=1

    oof2[oof2<0.5]=0

    oof_index = oof2[oof2 != y[oof2.index]].index

    

    # outlier로 판단 - replace

    train.iloc[oof_index, -1] = 1 - train.iloc[oof_index, -1]

    

    return train



def pseudo_labeling(preds, train, test, pseudo_label_thresold=0.99, rows_ratio=0.01):



    preds = pd.Series(preds)

    test = pd.DataFrame(test)

    

    preds = preds.sort_values()

    rows = np.round(len(preds)*rows_ratio, 0).astype(int)

    

    upper_index = preds[:rows].index

    lower_index = preds[-rows:].index

    

    # Pseudo labeling

#     preds = preds[(preds>pseudo_label_thresold) | (preds<1-pseudo_label_thresold)]

#     preds[preds>0.5] = 1

#     preds[preds<0.5] = 0

#     test = pd.concat([test.loc[preds.index], preds], axis=1)



    # using ratio

    _index = np.concatenate([upper_index, lower_index])

    preds = preds[_index]

    preds[preds>=0.5] = 1

    preds[preds<0.5] = 0

    test = pd.concat([test.loc[_index], preds], axis=1)



    # complete

    test.columns = train.columns

    train = pd.concat([train, test], ignore_index=True).reset_index(drop=True)

    

    return train



def gmm_fe(train, test, v_thres=V_THRES, seed=SEED):

    std_col = pd.concat([train, test]).std()

    feat_mask = std_col > v_thres       

    

    train1 = train.loc[:, feat_mask].values

    test1 = test.loc[:, feat_mask].values



    data = np.concatenate([train1, test1], axis=0)

    data2 = KernelPCA(n_components=data.shape[1], kernel='cosine', random_state=seed, n_jobs=1).fit_transform(data)

    

    c = AgglomerativeClustering(n_clusters=2)

    c_data = c.fit_predict(data).reshape(-1, 1)

    

    gmm_data1 = GaussianMixture(n_components=5, n_init=4, random_state=seed).fit(data2).predict_proba(data2)

    gmm_data2 = GaussianMixture(n_components=4, n_init=3, random_state=seed).fit(data2).predict_proba(data2)

    

    data2 = np.concatenate([data2, gmm_data1, gmm_data2, c_data], axis=1)

    data2 = StandardScaler().fit_transform(data2)

    

    train2 = data2[:train.shape[0]]

    test2 = data2[train.shape[0]:]

    

    return train2, test2
%%time

with single_threaded(np):

    oof, fake_oof, preds = execute_classifier(start_seed=SEED, seed_range=4, magic_range=10) # change (magic_range=512)
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)