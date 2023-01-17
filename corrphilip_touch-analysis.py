import sqlite3

import pandas as pd

import matplotlib as plt

import seaborn as sns

%matplotlib inline
touches_df = pd.read_csv("../input/touch.csv")
len(touches_df)
touches_df.head()
touches_df.tail()
touches_df.describe()
touches_df.dtypes
touches_df.isnull().values.any()
sns.distplot(touches_df['ZX']);
sns.distplot(touches_df['ZY']);
sns.distplot(touches_df['ZTIMESTAMP']);