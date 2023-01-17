import pandas as pd

import scipy.stats

import seaborn as sns

df = pd.read_csv("../input/20170308hundehalter.csv")

df.head(3)
alt = df["ALTER"].value_counts()

print(alt)

scipy.stats.chisquare(alt)
sns.countplot(df.ALTER, saturation=1, palette="cool", order=alt.index)
gesc = df["GESCHLECHT"].value_counts()

print(gesc)

scipy.stats.chisquare(gesc)
sns.countplot(df.GESCHLECHT, saturation=1, palette="cool", order=gesc.index)
table = pd.crosstab(df["ALTER"], df["GESCHLECHT"])

scipy.stats.chi2_contingency(table)