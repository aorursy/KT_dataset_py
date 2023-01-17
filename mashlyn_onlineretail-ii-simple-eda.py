import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')
df.shape
df.info()
df.describe()
df.corr()
from pandas.plotting import scatter_matrix

_ = scatter_matrix(df)