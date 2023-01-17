import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.mode.chained_assignment = None

plt.style.use('ggplot')
champs = pd.read_csv('../input/champs.csv')

champs.shape
champs.head()
matches = pd.read_csv('../input/matches.csv')

matches.creation = pd.to_datetime(matches.creation,unit="ms")

matches.platformid = matches.platformid.str[:-1]

print(matches.shape)

matches.head()
# drop cols

matches = matches[['id', 'platformid', 'seasonid', 'duration','creation']]
prefix = "../input/"

# matches = pd.read_csv(prefix+"matches.csv")

participants = pd.read_csv(prefix+"participants.csv")

stats = pd.read_csv(prefix+"stats1.csv", low_memory=False)

# stats2 = pd.read_csv(prefix+"stats2.csv", low_memory=False)

# stats = pd.concat([stats1,stats2])



# merge into a single DataFrame

a = pd.merge(participants, matches, left_on="matchid", right_on="id")

allstats = pd.merge(a, stats, left_on="matchid", right_on="id")

allstats = pd.merge(allstats, champs, left_on="championid", right_on="id").drop("championid",axis=1)
allstats.shapeall
allstats.head()
df = allstats.sample(n=100000)
df.drop(['id_x','id_y'],axis=1,inplace=True)

# df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))

# # df = pd.merge(df, champs, how = 'left', left_on = 'championid', right_on = 'id', suffixes=('', '_y'))



# # df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y'))

# df = pd.merge(df, matches[['id', 'duration','creation']], how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y')) 



# df.head()
df.columns
def final_position(row):

    if row['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):

        return row['role']

    else:

        return row['position']



df['adjposition'] = df.apply(final_position, axis = 1) 



df['team'] = df['player'].apply(lambda x: '1' if x <= 5 else '2')

df['team_role'] = df['team'] + ' - ' + df['adjposition']



# remove matchid with duplicate roles, e.g. 3 MID in same team, etc

remove_index = []

for i in ('1 - MID', '1 - TOP', '1 - DUO_SUPPORT', '1 - DUO_CARRY', '1 - JUNGLE', '2 - MID', '2 - TOP', '2 - DUO_SUPPORT', '2 - DUO_CARRY', '2 - JUNGLE'):

    df_remove = df[df['team_role'] == i].groupby('matchid').agg({'team_role':'count'})

    remove_index.extend(df_remove[df_remove['team_role']!=1].index.values)

    

# remove unclassified BOT, correct ones should be DUO_SUPPORT OR DUO_CARRY

remove_index.extend(df[df['adjposition'] == 'BOT']['matchid'].unique())

remove_index = list(set(remove_index))



print('# matches in dataset before cleaning: {}'.format(df['matchid'].nunique()))

df = df[~df['matchid'].isin(remove_index)]

print('# matches in dataset after cleaning: {}'.format(df['matchid'].nunique()))
df.shape
df.head()
df.columns
df.drop(['team_role'],axis=1,inplace=True)

df.drop_duplicates().to_csv("merged_flat_leagueGames_sample.csv.gz",index=False,compression="gzip")