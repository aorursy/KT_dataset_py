import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df1 = pd.read_csv('../input/all-the-news/articles1.csv')

df2 = pd.read_csv('../input/all-the-news/articles2.csv')

df3 = pd.read_csv('../input/all-the-news/articles3.csv')



df = pd.concat([df1, df2, df3])
npr_df = df[df['publication'] == 'NPR']

reuters_df = df[df['publication'] == 'Reuters']

washington_post_df = df[df['publication'] == 'Washington Post']

the_guardian_df = df[df['publication'] == 'Guardian']



all_df = df[df['publication'].isin(['NPR', 'Reuters', 'Washington Post', 'Guardian'])]



dfs = [npr_df, reuters_df, washington_post_df, the_guardian_df]
plt.bar(list(range(len(dfs))), [np.array([len(value) for value in df['content'].values]).mean() for df in dfs])
npr_df.to_csv('npr.csv')

reuters_df.to_csv('reuters.csv')

washington_post_df.to_csv('washington_post.csv')

the_guardian_df.to_csv('the_guardian.csv')



all_df.to_csv('all-the-news-subset.csv')