import sqlite3

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
glyph_df = pd.read_csv("../input/glyph.csv")
len(glyph_df)
glyph_df.head()
glyph_df.tail()
glyph_df.describe()
glyph_df.dtypes
glyph_df.isnull().values.any()

print(glyph_df["ZCLIENTHEIGHT"].unique())

print(glyph_df["ZCLIENTWIDTH"].unique())

print(glyph_df["ZDEVICE"].unique())

print(glyph_df["ZFINGER"].unique())

print(glyph_df["ZCHARACTER"].unique())
sns.distplot(glyph_df['ZDURATION']);