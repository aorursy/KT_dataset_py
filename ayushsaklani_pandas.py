from pandas import Series

temperatures = Series([23, 20, 25, 18])

temperatures
temperatures = Series([23, 20, 25, 18], index=['TOP', 'OUN', 'DAL', 'DEN'])

temperatures
temperatures['DAL']
temperatures[['DAL', 'OUN']]
temperatures[temperatures > 20]
temperatures + 2
dps = {'TOP': 14,

       'OUN': 18,

       'DEN': 9,

       'PHX': 11,

       'DAL': 23}



dewpoints = Series(dps)

dewpoints
'PHX' in dewpoints
'PHX' in temperatures
temperatures.name = 'temperature'

temperatures.index.name = 'station'
temperatures
from pandas import DataFrame



data = {'station': ['TOP', 'OUN', 'DEN', 'DAL'],

        'temperature': [23, 20, 25, 18],

        'dewpoint': [14, 18, 9, 23]}



df = DataFrame(data)

df
df['temperature']
df.dewpoint
df['wspeed'] = 0.

df
df.index = df.station

df
df.drop('station', 1, inplace=True)

df
df.loc['DEN']
df.T
df.values
df.temperature.values
import pandas as pd
df = pd.read_csv('../input/train_V2.csv')
df.head()
df.columns
len(df)
df.isnull().sum()
len(df)
df.head()
df.reset_index(drop=True, inplace=True)
df.head()
print('Min: {}\nMax: {}'.format(df.damageDealt.min(), df.damageDealt.max()))
df.teamKills.corr(df.damageDealt)
df.groupby('teamKills').mean()
df.groupby('teamKills').describe()