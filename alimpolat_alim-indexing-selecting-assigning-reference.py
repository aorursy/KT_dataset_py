import pandas as pd
reviews = pd.read_csv("../input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
df = reviews.copy()
df
df.country
df['country']
df['country'][0]
df.iloc[0]
reviews.iloc[:, 0]
reviews.iloc[:3, 0]
reviews.iloc[1:3, 0]
reviews.iloc[[0, 1, 2], 0]
reviews.iloc[-5:]
reviews.loc[0, 'country']
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
df.set_index("title")
df.country == 'Italy'
reviews.loc[df.country == 'Italy']
df.loc[(df.country == 'Italy') & (df.points >= 90)]
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
df.loc[df.country.isin(['Italy', 'France'])]
df.loc[df.country.isin(["Italy", "France"])]
df.loc[df.price.notnull()]
reviews['critic'] = 'everyone'
reviews['critic']

df['index_backwards'] = range(len(df), 0, -1)
df['index_backwards']
