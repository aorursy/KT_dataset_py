import pandas as pd 

import numpy as np



import os
train = pd.read_csv('../input/test_stage_2.tsv', delimiter = '\t').rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})



train = train.rename(columns={'A_Noun':'A', 'B_Noun':'B'})

train['A'] = pd.to_numeric(train['A'],  errors = 'ignore')

train['A'] = ((- np.random.randint(1, 12359, train.shape[0]) + 12359) / 2) / 10000

train['B'] = pd.to_numeric(train['B'],  errors = 'ignore')

train['B'] = ((- np.random.randint(1, 12359, train.shape[0]) + 12359) / 2) / 10000



sub = pd.read_csv('../input/sample_submission_stage_2.csv')

sub['A'] = sub['A'] - pd.to_numeric(train['A'],  errors = 'ignore')

sub['A'] = ((- np.random.randint(1, 12359, train.shape[0]) + 12359) / 2) / 10000

sub['B'] = sub['B'] - pd.to_numeric(train['B'],  errors = 'ignore')

sub['B'] = ((- np.random.randint(1, 12359, train.shape[0]) + 12359) / 2) / 10000

sub['NEITHER'] = ((10.0 - (train['A'] + train['B'])) / 6) / 10

sub[['ID', 'A', 'B', 'NEITHER']].to_csv("submission.csv", index=False)