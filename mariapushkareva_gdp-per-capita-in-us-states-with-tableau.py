import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/gdp-per-capita-in-us-states/bea-gdp-by-state.csv')
data.shape
data.head(60)
data.describe()