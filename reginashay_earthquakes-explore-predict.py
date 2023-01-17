import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/train.csv', nrows=10_000_000)

df.head()
df.time_to_failure.nunique()  # oh...
df.acoustic_data.unique()
sub = pd.read_csv('../input/sample_submission.csv')

sub.head()
sub.shape
test1 = pd.read_csv('../input/test/seg_00030f.csv')

test1.shape
test1.acoustic_data.nunique(), test1.acoustic_data.unique()
test1.head()
import os

test_files = list(os.listdir("../input/test"))

len(test_files), test_files[:5]
test1.acoustic_data.unique().mean()
%%time



folder = '../input/test/'

medians = []

test_files_feat = []



for file_path in test_files:

    test_files_feat.append(file_path[:-4])

    path = folder + file_path

    test_df = pd.read_csv(path)

    medians.append(int(test_df.acoustic_data.unique().mean()))
len(medians), medians[:5], len(test_files_feat), test_files_feat[:5]
test = pd.DataFrame({ 'seg_id' : test_files_feat, 'acoustic_data' : medians })

test.head()
from xgboost import XGBRegressor

#from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as mae
X, y = df.drop('time_to_failure', axis=1), df.time_to_failure
%%time

#rf = RandomForestRegressor(n_estimators=20)  #, criterion='mae')

#rf.fit(X, y)

xgb = XGBRegressor(max_depth=5, n_estimators=30, n_jobs=-1)

xgb.fit(X, y)
#rf.predict(test.drop('seg_id', axis=1))

xgb.predict(test.drop('seg_id', axis=1))
#test['time_to_failure'] = pd.Series(rf.predict(test.drop('seg_id', axis=1)))

test['time_to_failure'] = pd.Series(xgb.predict(test.drop('seg_id', axis=1)))

test.head()
subm = pd.merge(

    sub.drop('time_to_failure', axis=1),

    test.drop('acoustic_data', axis=1),

    on='seg_id'

)

subm.head()
subm.to_csv('submission.csv', index=False)