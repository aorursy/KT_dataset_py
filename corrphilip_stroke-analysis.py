import sqlite3

import pandas as pd

import matplotlib as plt

import seaborn as sns

%matplotlib inline
strokes_df = pd.read_csv("../input/stroke.csv")
len(strokes_df)
strokes_df.head()
strokes_df.tail()
strokes_df.describe()
strokes_df.dtypes
strokes_df.isnull().values.any()
strokes_df.duplicated(subset="ZINDEX", keep='first').unique()
sns.distplot(strokes_df['ZARCLENGTH']);
sns.distplot(strokes_df['ZDURATION']);