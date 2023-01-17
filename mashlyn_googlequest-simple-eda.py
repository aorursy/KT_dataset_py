import numpy as np

import pandas as pd

import os

import pandas_profiling as pp
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('../input/google-quest-challenge/train.csv')

df_test = pd.read_csv('../input/google-quest-challenge/test.csv')

df_sub = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
pp.ProfileReport(df_train)
pp.ProfileReport(df_test)