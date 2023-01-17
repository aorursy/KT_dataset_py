import pandas as pd

import numpy as np

df_hd=pd.read_csv('../input/heart.csv')

df_hd.columns
df_hd.head()
df_hd.dtypes
df_hd.describe()
df_hd.info()
df_hd.isnull().sum()