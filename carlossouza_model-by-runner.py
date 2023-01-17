import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
unique_runners = df['NflIdRusher'].nunique()

unique_plays = df['PlayId'].nunique()

print(f'There are {unique_runners} runners, responsible for {unique_plays} plays.')
agg_dic = {'PlayId': 'count', 'Yards': ['mean', 'std', 'min', 'max']}

dfp = df[['NflIdRusher', 'PlayId', 'Yards']].groupby('NflIdRusher').agg(agg_dic).sort_values(('PlayId', 'count'), ascending=False)

dfp
df_cum = dfp[[('PlayId', 'count')]].sort_values(('PlayId', 'count'), ascending=False)

df_cum.columns = ['count']

df_cum['cum'] = df_cum['count'].cumsum()

df_cum['perc'] = df_cum['cum'] / df_cum['cum'].iloc[-1]

df_cum.head(100)