%config IPCompleter.greedy=True





import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
import os

for i in [i for i in os.listdir(".") if "US" in i]:

    print(i)
df = pd.read_csv('../input/youtube-new/USvideos.csv')

df = df.head(10000)
df.info()
df.head()
travels = df[df['category_id'] == 19]

travels.head()
travels.groupby('channel_title').sum().sort_values('likes', ascending=False)['likes']
df.columns
tags = df['tags'].str.split('|').apply(pd.Series).reset_index().melt(id_vars='index').dropna().set_index('index')
df['tags'].str.split('|').apply(pd.Series).reset_index().melt(id_vars='index')