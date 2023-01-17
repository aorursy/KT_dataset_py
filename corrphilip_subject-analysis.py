import sqlite3

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
subject_df = pd.read_csv("../input/subject.csv")
len(subject_df)
subject_df.head()
subject_df.tail()
subject_df.describe()
subject_df.dtypes
subject_df.isnull().values.any()
print(subject_df["ZHANDEDNESS"].unique())

print(subject_df["ZNATIVELANGUAGE"].unique())

print(subject_df["ZSEX"].unique())
plt.figure(figsize = (12, 6))

ax = sns.countplot(x="ZAGE", data=subject_df, palette=sns.color_palette("Blues_d", n_colors=1, desat=.5))
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.countplot(x="ZHANDEDNESS", data=subject_df, ax=ax1)

sns.countplot(x="ZSEX", data=subject_df, ax=ax2)

plt.figure(figsize = (12, 6))

ax = sns.countplot(x="ZNATIVELANGUAGE", data=subject_df, palette=sns.color_palette("Blues_d", n_colors=1, desat=.5))
subject_df.to_csv("subject.csv", index=False)