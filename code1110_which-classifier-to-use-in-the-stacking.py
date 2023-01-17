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
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.covariance import GraphicalLasso

from sklearn.mixture import GaussianMixture

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.svm import LinearSVC



import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



print("Libraries were imported.")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print("Data were loaded.")

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
def get_mean_cov(x, y):

    model = GraphicalLasso()

    ones = (y==1).astype(bool)

    x2 = x[ones]

    model.fit(x2)

    p1 = model.precision_

    m1 = model.location_

    

    onesb = (y==0).astype(bool)

    x2b = x[onesb]

    model.fit(x2b)

    p2 = model.precision_

    m2 = model.location_

    

    return np.stack([m1, m2]), np.stack([p1, p2])
LRmodel1 = LogisticRegression()

LRmodel2 = RidgeClassifier()

LRmodel3 = LinearDiscriminantAnalysis()

LRmodel4 = LinearSVC()
oof1 = np.zeros(train.shape[0])

oof2 = np.zeros(train.shape[0])

oof3 = np.zeros(train.shape[0])

oof4 = np.zeros(train.shape[0])

predictions1 = np.zeros(test.shape[0])

predictions2 = np.zeros(test.shape[0])

predictions3 = np.zeros(test.shape[0])

predictions4 = np.zeros(test.shape[0])

for i in range(512):

        

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=1.5).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=5, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):

        

        # estimate mean and SD

        ms, ps = get_mean_cov(train3[train_index,:], train2.loc[train_index]['target'])

        

        # model fittings

        GMmodel = GaussianMixture(n_components=2, init_params='random', covariance_type='full',

                                tol=1e-3, reg_covar=0.001, max_iter=100, n_init=1,

                                means_init=ms, precisions_init=ps, verbose_interval=250)

        GMmodel.fit(np.concatenate([train3, test3], axis=0))

        QDAmodel = QuadraticDiscriminantAnalysis(0.5)

        QDAmodel.fit(train3[train_index,:], train2.loc[train_index]['target'])

        

        # prediction on test data

        GMpred = GMmodel.predict_proba(train3[test_index,:])[:, 0]

        QDApred = QDAmodel.predict_proba(train3[test_index,:])[:, 1]

        

        # stacking with different linear classifiers

        Stacked = np.vstack((GMpred, QDApred)).T

        LRmodel1.fit(Stacked, train2.loc[test_index]['target'])

        LRmodel2.fit(Stacked, train2.loc[test_index]['target'])

        LRmodel3.fit(Stacked, train2.loc[test_index]['target'])

        LRmodel4.fit(Stacked, train2.loc[test_index]['target'])

        oof1[idx1[test_index]] = LRmodel1.predict_proba(Stacked)[:,1]

        oof2[idx1[test_index]] = LRmodel2.predict(Stacked)

        oof3[idx1[test_index]] = LRmodel3.predict_proba(Stacked)[:,1]

        oof4[idx1[test_index]] = LRmodel3.predict(Stacked)

        

        # predictions with emsemble-stacking models

        TargetStacked = np.vstack((GMmodel.predict_proba(test3)[:, 0],

                                  QDAmodel.predict_proba(test3)[:, 1])).T

        predictions1[idx2] += LRmodel1.predict_proba(TargetStacked)[:,1] / skf.n_splits

        predictions2[idx2] += LRmodel2.predict(TargetStacked) / skf.n_splits

        predictions3[idx2] += LRmodel3.predict_proba(TargetStacked)[:,1] / skf.n_splits

        predictions4[idx2] += LRmodel4.predict(TargetStacked) / skf.n_splits

        

auc1 = roc_auc_score(train['target'], oof1)

auc2 = roc_auc_score(train['target'], oof2)

auc3 = roc_auc_score(train['target'], oof3)

auc4 = roc_auc_score(train['target'], oof4)

print('[Logistic Regression] ROC =',round(auc1, 5))

print('[Ridge Classifer] ROC =',round(auc2, 5))

print('[Linear Discriminant Analysis] ROC =',round(auc3, 5))

print('[Linear SVC] ROC =',round(auc4, 5))
aucs = [auc1, auc2, auc3, auc4]

predictions = [predictions1, predictions2, predictions3, predictions4]

submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = predictions[aucs.index(max(aucs))]

submission.to_csv('submission.csv', index=False)