# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle
from matplotlib import pyplot as plt
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%%time
dtype = pickle.load(open('../input/kdd2020-cpr/2018_dtypes.pkl', 'rb'))
del dtype['date']
df = pd.read_csv(
    '../input/kdd2020-cpr/2018.csv',
    dtype=dtype,
    parse_dates=['date']
)
%%time
print("Dropping nan columns:", df.dropna(axis=1).shape)
print("Dropping nan rows:", df.dropna(axis=0).shape)
nan_per_row = df.isna().sum(axis=1) / df.shape[1]
nan_per_line = df.isna().sum(axis=0) / df.shape[0]
nan_per_line
nan_per_row
plt.hist
plt.figure(1, figsize=(10, 6.6), dpi=150)
plt.hist(nan_per_line, bins=50)
plt.title('NaN percentage')
plt.show()