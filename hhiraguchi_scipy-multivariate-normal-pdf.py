import numpy as np

import pandas as pd

from scipy.stats import multivariate_normal

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape, test.shape)
%%time

MAGIC = 'wheezy-copper-turtle-magic'

cols = [c for c in train.columns if c not in ['id', 'target', MAGIC]]



oof_pdf = np.zeros(len(train))

preds_pdf = np.zeros(len(test))



for i in range(512):

    if i%20==0: print(i, end=' ')

    train2 = train[train[MAGIC]==i]

    test2 = test[test[MAGIC]==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])



    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, train2['target']):

        train4 = train3[train_index]

        target4 = train2['target'][train_index]



        mean0 = np.mean(train4[target4 == 0], axis=0)

        mean1 = np.mean(train4[target4 == 1], axis=0)

        cov0 = np.cov(train4[target4 == 0], rowvar=False)

        cov1 = np.cov(train4[target4 == 1], rowvar=False)

        

        pdf0 = multivariate_normal.pdf(train3[test_index], mean0, cov0)

        pdf1 = multivariate_normal.pdf(train3[test_index], mean1, cov1)

        oof_pdf[idx1[test_index]] = pdf1 / (pdf0 + pdf1)



        pdf0 = multivariate_normal.pdf(test3, mean0, cov0)

        pdf1 = multivariate_normal.pdf(test3, mean1, cov1)

        preds_pdf[idx2] += pdf1 / (pdf0 + pdf1) / skf.n_splits



print('fin')

print(roc_auc_score(train['target'], oof_pdf))
%%time

oof_pdf_2 = np.zeros(len(train))

preds_pdf_2 = np.zeros(len(test))



for i in range(512):

    if i%20==0: print(i, end=' ')

    train2 = train[train[MAGIC]==i]

    test2 = test[test[MAGIC]==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

    train3 = sel.transform(train2[cols])

    test3 = sel.transform(test2[cols])



    target = train2['target'].values.copy()

    p = 0.005

    target[oof_pdf[idx1] < p] = 0

    target[oof_pdf[idx1] > 1-p] = 1

    

    pred2 = preds_pdf[idx2]

    q = 0.01

    train3 = np.vstack([train3, test3[pred2 < q], test3[pred2 > 1-q]])

    target = np.hstack([target, np.zeros((pred2 < q).sum()), np.ones((pred2 > 1-q).sum())])

    

    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_index, test_index in skf.split(train3, target):

        train4 = train3[train_index]

        target4 = target[train_index]



        mean0 = np.mean(train4[target4 == 0], axis=0)

        mean1 = np.mean(train4[target4 == 1], axis=0)

        cov0 = np.cov(train4[target4 == 0], rowvar=False)

        cov1 = np.cov(train4[target4 == 1], rowvar=False)



        test_index = test_index[test_index < len(train2)]

        if len(test_index) > 0:

            pdf0 = multivariate_normal.pdf(train3[test_index], mean0, cov0)

            pdf1 = multivariate_normal.pdf(train3[test_index], mean1, cov1)

            oof_pdf_2[idx1[test_index]] += pdf1 / (pdf0 + pdf1)



        pdf0 = multivariate_normal.pdf(test3, mean0, cov0)

        pdf1 = multivariate_normal.pdf(test3, mean1, cov1)

        preds_pdf_2[idx2] += pdf1 / (pdf0 + pdf1) / skf.n_splits



print('fin')

print(roc_auc_score(train['target'], oof_pdf_2))
sample_submission = pd.read_csv('../input/sample_submission.csv')

sample_submission['target'] = preds_pdf_2

sample_submission.to_csv('submission_02.csv', index=False)
import matplotlib.pyplot as plt

plt.hist(preds_pdf_2, bins=100, log=True)

plt.grid()

plt.show()