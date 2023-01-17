%matplotlib inline

import os

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import datetime as dt

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns
plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

pd.set_option('display.max_columns', 99)

start = dt.datetime.now()
full = pd.read_csv('../input/0-feature-extraction/features_v3.csv.gz')

full.shape
train = full[full.IsTrain == 1].copy()

test = full[full.IsTrain == 0].copy()

test['known'] = 1 * test.Intersection.isin(train.Intersection)
overfitted = pd.read_csv('../input/3-xgboost-overfitting/overfitted.csv')

general = pd.read_csv('../input/4-xgboost-general/general.csv')
overfitted.shape, general.shape

subm = overfitted.merge(general, on='TargetId', suffixes=['Overfitted', 'General'])

subm.head()
subm['RowId'] = subm.TargetId.map(lambda s: int(s.split('_')[0]))

subm = subm.merge(test[['RowId', 'known']], on='RowId')

subm.head()

subm['Target'] = subm.TargetOverfitted * subm.known + subm.TargetGeneral * (1 - subm.known)
test.known.mean()
subm[subm.known == 1].head(12)

subm[subm.known == 0].head(12)

subm.shape
subm.min()

subm.Target = subm.Target.clip(0, None)
subm[['TargetId', 'Target']].to_csv('submission_combined.csv', index=False)
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))