import numpy as np

import pandas as pd

import os

import pandas_profiling as pp
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/abalone-dataset/abalone.csv')
pp.ProfileReport(df)