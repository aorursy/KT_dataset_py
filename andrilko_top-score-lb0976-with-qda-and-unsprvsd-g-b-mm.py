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
def permute_predict(y):

    _y = y.copy()

    _c1 = _y < 0.00001

    _c2 = _y > 0.99999

    _y[_c1] = _y[_c1].max() - _y[_c1] + _y[_c1].min()

    _y[_c2] = _y[_c2].max() - _y[_c2] + _y[_c2].min()

    return _y
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.mixture import GaussianMixture as GMM

from sklearn.mixture import BayesianGaussianMixture as BGM

from tqdm import tqdm_notebook

from sklearn.covariance import GraphicalLasso



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')
oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(2.3).fit_transform(data[cols])

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    for c in range(train3.shape[1]):

        low_=np.quantile(train3[:,c] , 0.001)

        up_=np.quantile(train3[:,c], 0.999)

        train3[:,c]=np.clip(train3[:,c],low_, up_ )

        test3[:,c]=np.clip(test3[:,c],low_, up_ )

        

#     train3 = ((train3) / data[:train2.shape[0]].std(axis=1)[:, np.newaxis])

#     test3 = ((test3) / data[train2.shape[0]:].std(axis=1)[:, np.newaxis])

    

    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):

        gmm=GMM(n_components=5, random_state=42, covariance_type='full')

        gmm.fit(np.vstack([train3[train_index], test3]))

        gmm_1_train=gmm.predict_proba(train3[train_index])

        gmm_1_val=gmm.predict_proba(train3[test_index])

        gmm_1_test=gmm.predict_proba(test3)



        gmm=GMM(n_components=4, random_state=42, covariance_type='full')

        gmm.fit(np.vstack([train3[train_index], test3]))

        gmm_2_train=gmm.predict_proba(train3[train_index])

        gmm_2_val=gmm.predict_proba(train3[test_index])

        gmm_2_test=gmm.predict_proba(test3)



        gmm=GMM(n_components=6, random_state=42, covariance_type='full')

        gmm.fit(np.vstack([train3[train_index], test3]))

        gmm_3_train=gmm.predict_proba(train3[train_index])

        gmm_3_val=gmm.predict_proba(train3[test_index])

        gmm_3_test=gmm.predict_proba(test3)







        bgm=BGM(n_components=5, random_state=42)

        bgm.fit(np.vstack([train3[train_index], test3]))

        bgm_1_train=bgm.predict_proba(train3[train_index])

        bgm_1_val=bgm.predict_proba(train3[test_index])

        bgm_1_test=bgm.predict_proba(test3)



        bgm=BGM(n_components=4, random_state=42)

        bgm.fit(np.vstack([train3[train_index], test3]))

        bgm_2_train=bgm.predict_proba(train3[train_index])

        bgm_2_val=bgm.predict_proba(train3[test_index])

        bgm_2_test=bgm.predict_proba(test3)



        bgm=BGM(n_components=6, random_state=42)

        bgm.fit(np.vstack([train3[train_index], test3]))

        bgm_3_train=bgm.predict_proba(train3[train_index])

        bgm_3_val=bgm.predict_proba(train3[test_index])

        bgm_3_test=bgm.predict_proba(test3)

    

        _train = np.hstack((train3[train_index],

                            gmm_1_train, gmm_2_train, gmm_3_train,

                            bgm_1_train, bgm_2_train, bgm_3_train))

        _val = np.hstack((train3[test_index],

                            gmm_1_val, gmm_2_val, gmm_3_val,

                            bgm_1_val, bgm_2_val, bgm_3_val))

        _test = np.hstack((test3,

                            gmm_1_test, gmm_2_test, gmm_3_test,

                            bgm_1_test, bgm_2_test, bgm_3_test))

        clf = QuadraticDiscriminantAnalysis(reg_param=0.04, tol=0.01) #0.04 bst - 0.9734+

        clf.fit(_train,train2.loc[train_index]['target'])

        

        oof[idx1[test_index]] = clf.predict_proba(_val)[:,1]

        preds[idx2] += clf.predict_proba(_test)[:,1] / skf.n_splits

    print(i, roc_auc_score(train2['target'], oof[idx1]))

print(roc_auc_score(train['target'], oof))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = permute_predict(preds)

sub.to_csv('submission.csv',index=False)



import matplotlib.pyplot as plt

plt.hist(preds,bins=100)

plt.title('Final Test.csv predictions')

plt.show()