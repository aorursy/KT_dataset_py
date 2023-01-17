import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



import os
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test  = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sub   = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')



train.shape, test.shape, sub.shape
train['sex'] = train['sex'].fillna('na')

train['age_approx'] = train['age_approx'].fillna(0)

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('na')



test['sex'] = test['sex'].fillna('na')

test['age_approx'] = test['age_approx'].fillna(0)

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('na')
train.head()
test.head()
L = 15

feat = ['sex','age_approx','anatom_site_general_challenge']



M = train.target.mean()

te = train.groupby(feat)['target'].agg(['mean','count']).reset_index()

te['ll'] = ((te['mean']*te['count'])+(M*L))/(te['count']+L)

del te['mean'], te['count']



test = test.merge( te, on=feat, how='left' )

test['ll'] = test['ll'].fillna(M)



test.head()
sub.target = test.ll.values

sub.head(10)
sub.to_csv( 'submission.csv', index=False )