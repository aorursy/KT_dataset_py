import pandas as pd

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

plt.rcParams['figure.figsize'] = [15,15]
df = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
df.head(10)
df.tail(10)
print(list(df.columns))
print("No. of rows: ", len(df))
df['country'].dropna(inplace = True)

print(dict(Counter(df['country'].values).most_common(5)))
df_movie = df[df['type'] =='Movie']

print(set(df_movie['duration']))
df_movie['duration'] = df_movie['duration'].map(lambda d: d.rstrip('min')).astype(int)

print(set(df_movie['duration']))
keys = []

for i in dict(Counter(df_movie['listed_in'].values).most_common(5)):

    keys.append(i)

print(keys)

df_new = df_movie[df_movie['listed_in'].isin(keys)]

sns.boxplot(x = df_new['listed_in'], y = df_new['duration'])
df_movie['duration'].hist(bins=100)