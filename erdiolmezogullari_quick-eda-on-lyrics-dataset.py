import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/lyrics.csv')
df.head()
df.info()
df.dropna(inplace=True)
assert df.groupby(['year']).aggregate('count')['index'].sum() == len(df)
df_tmp = df.groupby(['genre']).aggregate({
    'genre' : 'count'
})['genre'].sort_values()

display(df_tmp.describe())
df_tmp.plot.bar()
df.groupby(['year']).aggregate({
    'year' : 'count'
})['year'].plot.bar()
def times(c):
    if 1950 <= c and c < 1960:
        return 1950
    elif 1960 <= c and c < 1970:
        return 1960
    if 1970 <= c and c < 1980:
        return 1970
    elif 1980 <= c and c < 1990:
        return 1980
    elif 1990 <= c and c < 2000:
        return 1990
    elif 2000 <= c and c < 2010:
        return 2000
    elif 2010 <= c and c < 2020:
        return 2010
    return -1

df['year'] = df['year'].apply(int, 1)
df['times'] = df['year'].apply(times)
df.head()
df[df['times'] == -1]
idx = df[df['times'] == -1].index
assert len(df.drop(index=idx)) ==  (len(df) - len(idx))
df = df.drop(index=idx).reset_index().drop(['index', 'level_0'], 1)
df.groupby(['times']).aggregate({
    'times' : 'count'
}).sort_index(ascending=True)['times'].plot.bar()
df_artist = df.groupby(['artist']).aggregate({
    'artist' : ['count']
})
df_artist = df_artist.sort_values([('artist','count')])

display(df_artist.describe())
display(df_artist[df_artist[('artist','count')] >= 100].describe())
display(df_artist[df_artist[('artist','count')] >= 500].describe())
df_artist[df_artist[('artist','count')] >= 300][('artist','count')].plot.bar()
df.pivot_table(values=['year'], index=['genre'], columns=['times'], aggfunc='count', fill_value=0)
df = df[df['times'] != 1960]
df = df[df['genre'] != 'Not Available']

df.pivot_table(values=['year'], index=['genre'], columns=['times'], aggfunc='count', fill_value=0)
# df_tmp = df.pivot_table(values=['year'], index=['artist'], columns=['times'], aggfunc='count', fill_value=0)
# df_tmp

# df_tmp = df.pivot_table(values=['year'], index=['artist'], columns=['genre'], aggfunc='count', fill_value=0)
# df_tmp

df
df['l_len'] = df['lyrics'].apply(lambda x : len(x), 0)
df['token_n'] = df['']
df_test = df[df['genre'] == 'Not Available'][['lyrics']]
df_train = df[df['genre'] != 'Not Available'][['lyrics', 'genre']]
