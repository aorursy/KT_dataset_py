import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
plt.style.use('ggplot')
%matplotlib inline
champs = pd.read_csv("../input/champs.csv")
champs.head()
matches = pd.read_csv("../input/matches.csv")
matches.head()
participants = pd.read_csv('../input/participants.csv')
participants.tail()
stats1 = pd.read_csv('../input/stats1.csv')
stats1.head(2)
stats2 = pd.read_csv('../input/stats2.csv')
stats2.head(2)
stats = stats1.append(stats2)
stats.shape
stats.head()
df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))

df = pd.merge(df , champs, how = 'left', left_on= 'championid', right_on='id'
             ,suffixes=('', '_y') )

df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id'
              , suffixes=('', '_y'))
df.columns
def final_position(col):
    if col['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):
        return col['role']
    else:
        return col['position']
df['adjposition'] = df.apply(final_position, axis = 1)
df.head()
df['team'] = df['player'].apply(lambda x: '1' if x <= 5 else '2')
df['team_role'] = df['team'] + ' - ' + df['adjposition']
df.head()
remove_index = []
for i in ('1 - MID', '1 - TOP', '1 - DUO_SUPPORT', '1 - DUO_CARRY', '1 - JUNGLE',
          '2 - MID', '2 - TOP', '2 - DUO_SUPPORT', '2 - DUO_CARRY', '2 - JUNGLE'):
    df_remove = df[df['team_role'] == i].groupby('matchid').agg({'team_role':'count'})
    remove_index.extend(df_remove[df_remove['team_role'] != 1].index.values)
remove_index.extend(df[df['adjposition'] == 'BOT']['matchid'].unique())
remove_index = list(set(remove_index))
print('# matches in dataset before cleaning:{}'.format(df['matchid'].nunique()))
df = df[~df['matchid'].isin(remove_index)]
print('# matches in dataset after cleaning: {}'.format(df['matchid'].nunique()))
df.columns
df = df[['id', 'matchid', 'player', 'name', 'adjposition', 'team_role',
         'win', 'kills', 'deaths', 'assists', 'turretkills','totdmgtochamp',
         'totheal', 'totminionskilled', 'goldspent', 'totdmgtaken', 'inhibkills',
         'pinksbought', 'wardsplaced', 'duration', 'platformid',
         'seasonid', 'version']]
df.head()
df_v = df.copy()
# Putting ward limits
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x<30 else 30)
df_v['wardsplaced'] = df_v['wardsplaced'].apply(lambda x: x if x>0 else 0)

df_v['wardsplaced'].head()
plt.figure(figsize=(12,10))
sns.violinplot(x='seasonid', y= 'wardsplaced', hue='win', data= df_v, split = True
              , inner= 'quartile')
plt.title('Wardsplaced by season : win & lose')
df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

m = np.zeros_like(df_corr.corr(), dtype=np.bool)
m[np.triu_indices_from(m)] = True

plt.figure(figsize=(16,10))
sns.heatmap(df_corr.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f',
            linewidths=.5, mask = m)

plt.title('Correlations - win vs factors (all games)')

df_corr_2 = df._get_numeric_data()
# for games less than 25mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]
df_corr_2 = df_corr_2.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

m = np.zeros_like(df_corr_2.corr(), dtype=np.bool)
m[np.triu_indices_from(m)] = True

plt.figure(figsize=(16,10))
sns.heatmap(df_corr_2.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f',
            linewidths=.5, mask = m)

plt.title('Correlations - win vs factors (for games last less than 25 mins)')
df_corr_3 = df._get_numeric_data()
# for games more than 40mins
df_corr_3 = df_corr_3[df_corr_3['duration'] > 2400]
df_corr_3 = df_corr_3.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)

m = np.zeros_like(df_corr_3.corr(), dtype=np.bool)
m[np.triu_indices_from(m)] = True

plt.figure(figsize=(16,10))
sns.heatmap(df_corr_3.corr(), cmap = 'coolwarm', annot= True, fmt = '.2f',
            linewidths=.5, mask = m)

plt.title('Correlations - win vs factors (for games last less than 40 mins)')
pd.options.display.float_format = '{:,.1f}'.format


df_win_rate = df.groupby('name').agg({'win': 'sum','name': 'count',
                                     'kills':'mean','deaths':'mean',
                                     'assists':'mean'})
df_win_rate.columns = ['win' , 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win'] / df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate',ascending= False)
df_win_rate = df_win_rate[['total matches', 'win rate' , 'K' , 'D', 'A', 'KDA']]


print('Top 10 win rate')
print(df_win_rate.head(10))
print('Least 10 win rate')
print(df_win_rate.tail(10))
df_win_rate.reset_index(inplace= True)
# plotting the result visually
plt.figure(figsize=(16,30))
cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="win rate", y="name", hue='KDA',
                     palette=cmap, sizes=(10, 200),
                     data=df_win_rate)
df_win_rate.head()
df_2 = df.sort_values(['matchid','adjposition'], ascending = [1,1])

df_2['shift 1'] = df_2['name'].shift()
df_2['shift -1'] = df_2['name'].shift(-1)

def matchup(x):
    if x['player'] <= 5:
        if x['name'] < x['shift -1']:
            name_return = x['name'] + ' vs ' + x['shift -1']
        else:
            name_return = x['shift -1'] + ' vs ' + x['name']
    else:
        if x['name'] < x['shift 1']:
            name_return = x['name'] + ' vs ' + x['shift 1']
        else:
            name_return = x['shift 1'] + ' vs ' + x['name']
    return name_return

df_2['matchup'] = df_2.apply(matchup, axis = 1)
df_2['win_adj'] = df_2.apply(lambda x: x['win'] if x['name'] == x['matchup'].split(' vs')[0]
                            else 0, axis = 1)

df_2.head()
df_matchup = df_2.groupby(['adjposition', 'matchup']).agg({'win_adj': 'sum', 'matchup': 'count'})
df_matchup.columns = ['win matches', 'total matches']
df_matchup['total matches'] = df_matchup['total matches'] / 2
df_matchup['win rate'] = df_matchup['win matches'] /  df_matchup['total matches']  * 100
df_matchup['dominant score'] = df_matchup['win rate'] - 50
df_matchup['dominant score (ND)'] = abs(df_matchup['dominant score'])
df_matchup = df_matchup[df_matchup['total matches'] > df_matchup['total matches'].sum()*0.0001]

df_matchup = df_matchup.sort_values('dominant score (ND)', ascending = False)
df_matchup = df_matchup[['total matches', 'dominant score']]                   
df_matchup = df_matchup.reset_index()

print('Dominant score +/- means first/second champion dominant:')

for i in df_matchup['adjposition'].unique(): 
        print('\n{}:'.format(i))
        print(df_matchup[df_matchup['adjposition'] == i].iloc[:,1:].head(5))
df_matchup['adjposition'].unique()

df_matchup_TOP = df_matchup.loc[df_matchup['adjposition'] == 'TOP']
df_matchup_JUNGLE = df_matchup.loc[df_matchup['adjposition'] == 'JUNGLE']
df_matchup_MID = df_matchup.loc[df_matchup['adjposition'] == 'MID']
df_matchup_DUO_CARRY = df_matchup.loc[df_matchup['adjposition'] == 'DUO_CARRY']
df_matchup_DUO_SUPPORT = df_matchup.loc[df_matchup['adjposition'] == 'DUO_SUPPORT']


print(df_matchup_TOP.shape)
print(df_matchup_JUNGLE.shape)
print(df_matchup_MID.shape)
print(df_matchup_DUO_CARRY.shape)
print(df_matchup_DUO_SUPPORT.shape)
# plotting duo carry 
plt.figure(figsize=(16,60))
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_DUO_CARRY,
            label="Total", color="b")
# plotting TOP

plt.figure(figsize=(16,200))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_TOP,
            label="Total", color="c")
# plotting jungle

plt.figure(figsize=(16,100))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_JUNGLE,
            label="Total", color="g")
# plotting mid

plt.figure(figsize=(16,100))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_MID,
            label="Total", color="r")
# plotting support

plt.figure(figsize=(16,100))
sns.set()
sns.set_color_codes("dark")
sns.barplot(x="dominant score", y="matchup", data=df_matchup_DUO_SUPPORT,
            label="Total", color="m")