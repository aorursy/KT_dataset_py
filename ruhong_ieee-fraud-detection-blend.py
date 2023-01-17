import os

import random

import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
SEED = 31
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

seed_everything(SEED)
template = '../input/{}/submission.csv'



files = [

    template.format('ieee-fraud-detection-lgb'),

    template.format('ieee-fraud-detection-xgb'),

    #template.format('ieee-fraud-detection-rf'),

    #template.format('ieee-fraud-detection-rf-leafwise'),

    #template.format('ieee-fraud-detection-et'),

    #template.format('ieee-fraud-detection-et-leafwise'),

    #template.format('ieee-fraud-detection-logistic')

]



print(f'len(files)={len(files)}')
def get_median(files):

    outs = [pd.read_csv(f, index_col=0) for f in files]

    concat_sub = pd.concat(outs, axis=1, sort=True)

    preds = concat_sub.median(axis=1).values

    return preds





preds = get_median(files)

print(f'len(preds)={len(preds)}')
sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')

sub['isFraud'] = preds

sub.head()
sub.to_csv('submission.csv', index=False)

print(os.listdir("."))