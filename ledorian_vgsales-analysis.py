import pandas as pd

import sklearn as sk

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/vgsales.csv')
df.head()
df.dtypes
col_quan = df.select_dtypes(exclude = ['object']).columns

col_qual = df.select_dtypes(include = ['object']).columns
df_qual = df[col_qual]
print("Number of unique values in each qualitative variable")

print("All:  {}".format(len(df)))

for col in df_qual.columns:

    nb_unique = len(df[col].unique())

    print("{}: {}".format(col[:4], nb_unique))
print(len(df[df[['Name']].duplicated()]))

print(len(df[df[['Name', 'Platform']].duplicated()]))

print(len(df[df[['Name', 'Platform', 'Year']].duplicated()]))

df[df[['Name', 'Platform', 'Year']].duplicated(keep=False)]
df = df.sort_values(['Global_Sales'], ascending=False)

df = df[~df[['Name', 'Platform', 'Year']].duplicated(keep='first')]
fig, ax = plt.subplots()

df_qual['Platform'].value_counts().plot(ax=ax, kind='bar')

plt.show()
df.describe()
df[df.isnull().any(axis=1)]