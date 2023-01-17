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
import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split

from tqdm import tqdm_notebook

import warnings

import multiprocessing

from scipy.optimize import minimize  

warnings.filterwarnings('ignore')
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

print(train.shape, test.shape)
reg_params = [0.01,0.23,0.01,0.08,0.27,0.03,0.16,0.33,0.28,0.02,0.01,0.07,0.01,0.31

,0.28,0.26,0.02,0.1,0.11,0.17,0.42,0.34,0.11,0.22,0.61,0.12,0.14,0.19

,0.02,0.01,0.02,0.06,0.36,0.09,0.11,0.01,0.03,0.11,0.01,0.11,0.23,0.02

,0.17,0.14,0.12,0.06,0.08,0.01,0.26,0.01,0.02,0.04,0.33,0.03,0.07,0.16

,0.01,0.02,0.28,0.17,0.04,0.19,0.05,0.11,0.03,0.22,0.17,0.16,0.06,0.14

,0.01,0.07,0.05,0.06,0.2,0.05,0.02,0.09,0.05,0.01,0.08,0.1,0.01,0.22

,0.19,0.13,0.02,0.03,0.12,0.07,0.15,0.09,0.02,0.16,0.01,0.39,0.12,0.14

,0.09,0.41,0.32,0.13,0.15,0.01,0.41,0.12,0.01,0.13,0.17,0.03,0.08,0.12

,0.21,0.01,0.16,0.28,0.01,0.02,0.11,0.21,0.27,0.03,0.17,0.05,0.02,0.4

,0.28,0.01,0.03,0.28,0.02,0.12,0.17,0.1,0.01,0.1,0.01,0.24,0.11,0.18

,0.17,0.08,0.1,0.04,0.13,0.04,0.12,0.25,0.13,0.19,0.05,0.03,0.1,0.01

,0.05,0.01,0.01,0.01,0.18,0.24,0.13,0.01,0.15,0.01,0.09,0.04,0.06,0.01

,0.06,0.11,0.21,0.08,0.21,0.04,0.04,0.04,0.2,0.01,0.19,0.14,0.11,0.03

,0.23,0.01,0.04,0.13,0.46,0.04,0.07,0.08,0.07,0.01,0.11,0.2,0.07,0.23

,0.2,0.14,0.07,0.06,0.16,0.02,0.33,0.13,0.11,0.06,0.22,0.22,0.19,0.03

,0.1,0.37,0.22,0.01,0.01,0.22,0.07,0.01,0.23,0.35,0.03,0.29,0.01,0.04

,0.01,0.04,0.07,0.23,0.2,0.09,0.01,0.23,0.3,0.06,0.09,0.04,0.46,0.25

,0.14,0.01,0.04,0.03,0.01,0.04,0.11,0.08,0.01,0.09,0.05,0.1,0.05,0.28

,0.02,0.08,0.01,0.06,0.38,0.04,0.01,0.15,0.21,0.01,0.01,0.45,0.18,0.27

,0.24,0.01,0.04,0.14,0.13,0.18,0.22,0.32,0.13,0.07,0.26,0.17,0.12,0.14

,0.09,0.13,0.08,0.09,0.07,0.01,0.02,0.04,0.01,0.07,0.32,0.01,0.36,0.09

,0.11,0.06,0.46,0.11,0.16,0.21,0.01,0.1,0.01,0.1,0.23,0.05,0.33,0.01

,0.24,0.04,0.01,0.04,0.1,0.01,0.36,0.44,0.03,0.08,0.21,0.01,0.18,0.01

,0.17,0.19,0.03,0.01,0.18,0.15,0.48,0.06,0.17,0.18,0.37,0.01,0.31,0.01

,0.16,0.18,0.11,0.08,0.08,0.07,0.28,0.02,0.09,0.08,0.01,0.09,0.01,0.07

,0.01,0.24,0.09,0.02,0.37,0.16,0.04,0.14,0.22,0.06,0.29,0.16,0.06,0.06

,0.04,0.05,0.25,0.07,0.01,0.01,0.21,0.02,0.04,0.3,0.39,0.02,0.23,0.22

,0.05,0.01,0.06,0.05,0.02,0.01,0.02,0.12,0.11,0.05,0.01,0.2,0.01,0.08

,0.08,0.04,0.33,0.06,0.16,0.35,0.18,0.13,0.01,0.01,0.12,0.18,0.01,0.01

,0.17,0.08,0.48,0.18,0.01,0.02,0.35,0.15,0.34,0.01,0.14,0.01,0.32,0.34

,0.15,0.1,0.18,0.18,0.11,0.24,0.01,0.13,0.03,0.36,0.01,0.08,0.01,0.13

,0.07,0.08,0.24,0.01,0.05,0.05,0.1,0.07,0.21,0.01,0.08,0.11,0.09,0.04

,0.01,0.11,0.02,0.09,0.16,0.51,0.17,0.09,0.03,0.12,0.06,0.18,0.01,0.01

,0.02,0.02,0.01,0.27,0.28,0.09,0.02,0.13,0.03,0.16,0.15,0.04,0.17,0.19

,0.26,0.01,0.04,0.1,0.04,0.01,0.07,0.05,0.02,0.59,0.01,0.2,0.12,0.1

,0.07,0.18,0.08,0.01,0.34,0.15,0.28,0.12,0.05,0.01,0.11,0.08,0.12,0.04

,0.05,0.33,0.01,0.01,0.09,0.07,0.19,0.09]
oof = np.zeros(len(train))

preds = np.zeros(len(test))

#reg_params = np.zeros(512)

for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5)), ('scaler', StandardScaler())])

    data2 = pipe.fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=30, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(reg_params[i])

#         qda = QuadraticDiscriminantAnalysis()

#         clf = GridSearchCV(qda, params, cv=4)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

#         reg_params[i] = clf.best_params_['reg_param']

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')

# print('best_params:', reg_params)
for i in range(5):

    test['target'] = preds

    test.loc[test['target'] > 0.94, 'target'] = 1

    test.loc[test['target'] < 0.06, 'target'] = 0

    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

    print(usefull_test.shape[0])

    new_train.loc[oof > 0.98, 'target'] = 1

    new_train.loc[oof < 0.02, 'target'] = 0

    oof2 = np.zeros(len(train))

    preds = np.zeros(len(test))

    for i in tqdm_notebook(range(512)):

        train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx1 = train[train['wheezy-copper-turtle-magic']==i].index

        idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        pipe = Pipeline([('vt', VarianceThreshold(threshold=1.5)), ('scaler', StandardScaler())])

        data2 = pipe.fit_transform(data[cols])

        train3 = data2[:train2.shape[0]]

        test3 = data2[train2.shape[0]:]

        skf = StratifiedKFold(n_splits=30, random_state=42)

        for train_index, test_index in skf.split(train2, train2['target']):

            oof_test_index = [t for t in test_index if t < len(idx1)]

            clf = QuadraticDiscriminantAnalysis(reg_params[i])

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            if len(oof_test_index) > 0:

                oof2[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]

            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

    auc = roc_auc_score(train['target'], oof2)

    print(f'AUC: {auc:.5}')
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)