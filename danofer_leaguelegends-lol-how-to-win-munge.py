import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.options.mode.chained_assignment = None

plt.style.use('ggplot')
champs = pd.read_csv('../input/champs.csv')

champs.shape
matches = pd.read_csv('../input/matches.csv')

matches.creation = pd.to_datetime(matches.creation,unit="ms")

matches.platformid = matches.platformid.str[:-1]

print(matches.shape)

matches.head()
# drop cols

matches = matches[['id', 'platformid', 'seasonid', 'duration','creation']]
participants = pd.read_csv('../input/participants.csv')

participants.shape
participants.head()
# # Take only first 1 million rows / stats1

# stats = pd.read_csv('../input/stats1.csv')

# print(stats.shape)

# stats.head()
champs.head()
stats1 = pd.read_csv('../input/stats1.csv')

stats1.shape

stats2 = pd.read_csv('../input/stats2.csv')

stats2.shape



stats = stats1.append(stats2)

stats.shape
df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))

# df = pd.merge(df, champs, how = 'left', left_on = 'championid', right_on = 'id', suffixes=('', '_y'))



# df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y'))

df = pd.merge(df, matches[['id', 'duration','creation']], how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y')) 



df.head()
df.columns
def final_position(row):

    if row['role'] in ('DUO_SUPPORT', 'DUO_CARRY'):

        return row['role']

    else:

        return row['position']



df['adjposition'] = df.apply(final_position, axis = 1) 



df['team'] = df['player'].apply(lambda x: '0' if x <= 5 else '1')

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
df = df[['id', 'matchid', 'player', 'name', 'adjposition', 'team_role', 'win', 'kills', 'deaths', 'assists', 'turretkills','totdmgtochamp', 'totheal', 'totminionskilled', 'goldspent', 'totdmgtaken', 'inhibkills', 'pinksbought', 'wardsplaced', 'duration', 'platformid', 'seasonid', 'version']]

df.head(10)
df_corr = df._get_numeric_data()

df_corr = df_corr.drop(['id', 'matchid', 'player', 'seasonid'], axis = 1)



mask = np.zeros_like(df_corr.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(10, 150, as_cmap=True)



plt.figure(figsize = (15,10))

sns.heatmap(df_corr.corr(), cmap = cmap, annot = True, fmt = '.2f', mask = mask, square=True, linewidths=.5, center = 0)

plt.title('Correlations - win vs factors (all games)')
pd.options.display.float_format = '{:,.1f}'.format



df_win_rate_role = df.groupby(['name','adjposition']).agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})

df_win_rate_role.columns = ['win matches', 'total matches', 'K', 'D', 'A']

df_win_rate_role['win rate'] = df_win_rate_role['win matches'] /  df_win_rate_role['total matches'] * 100

df_win_rate_role['KDA'] = (df_win_rate_role['K'] + df_win_rate_role['A']) / df_win_rate_role['D']

df_win_rate_role = df_win_rate_role.sort_values('win rate', ascending = False)

df_win_rate_role = df_win_rate_role[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]



# occur > 0.01% of matches

df_win_rate_role = df_win_rate_role[df_win_rate_role['total matches'] > df_win_rate_role['total matches'].sum()*0.0001]

print('Top 10 win rate with role (occur > 0.01% of total # matches)')

print(df_win_rate_role.head(10))

print('Bottom 10 win rate with role (occur > 0.01% of total # matches)')

print(df_win_rate_role.tail(10))
df_matchup = df_2.groupby(['adjposition', 'match up']).agg({'win_adj': 'sum', 'match up': 'count'})

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
df['name'].unique()
# remove missing data



df_4 = df_4.dropna()



y = df_4['T1 win']

X = df_4[df_4.columns.difference(['T1 win'])]



# prepare for prediction, below is a team comp I am curious about

s = pd.Series(['KogMaw', 'Leona', 'Lee Sin', 'Ahri', 'Riven', 'Kalista', 'Blitzcrank', 'Xin Zhao', 'Ryze', 'Shen', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)



# prepare for SKT vs SSG, no KR server available so I use NA as reference

s_g1 = pd.Series(['Varus', 'Lulu', 'Gragas', 'Cassiopeia', 'Gnar', 'Xayah', 'Janna', 'Zac', 'Malzahar', 'Kennen', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)

s_g2 = pd.Series(['Varus', 'Lulu', 'Gragas', 'Ryze', 'Yasuo', 'Xayah', 'Janna', 'Jarvan IV', 'Malzahar', 'Gnar', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)

s_g3 = pd.Series(['Tristana', 'Leona', 'Gragas', 'Karma', 'Trundle', 'Varus', 'Lulu', 'Sejuani', 'Malzahar', 'ChoGath', 'NA1', 8], index = df_4[df_4.columns.difference(['T1 win'])].columns)





# add for transformation

X = X.append(s, ignore_index=True)

X = X.append(s_g1, ignore_index=True)

X = X.append(s_g2, ignore_index=True)

X = X.append(s_g3, ignore_index=True)



# label string to numeric

le_t = X.apply(le.fit)

X_t_1 = X.apply(le.fit_transform)



enc = preprocessing.OneHotEncoder()

enc_t = enc.fit(X_t_1)

X_t_2 = enc_t.transform(X_t_1)



# split train & test, exclude last row (our curious comp)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_t_2[:-4], y, random_state=0)



# category with just 0 / 1, no magnitude meaning in category like above approach

print('Accuracy on dataset converted from label to binary category (with platform id, seasonid):')



clf_lr = LogisticRegression(random_state = 0).fit(X_train_2, y_train_2)

acc_lr = clf_lr.score(X_test_2, y_test_2)

print('logistic regression : {}'.format(acc_lr))



clf_bnb = BernoulliNB().fit(X_train_2, y_train_2)

acc_bnb = clf_bnb.score(X_test_2, y_test_2)

print('naive bayes : {}'.format(acc_bnb))



clf_xb = xgboost.XGBClassifier(random_state = 0).fit(X_train_2, y_train_2)

acc_xb = clf_xb.score(X_test_2, y_test_2)

print('xgboost : {}'.format(acc_xb))
# predict certain team comp win rate

print('Team 1 win rate for below team comp: {} \n{}'.format(clf_lr.predict_proba(X_t_2[-4])[0][1], pd.DataFrame(s, columns = ['Champ'])))

# predict SKT vs SSG 2017 World Final win rate

print('Team 1 = SKT / Team 2 = SSG')

print('Game 1 SKT win rate: {} \n{}\n'.format(clf_lr.predict_proba(X_t_2[-3])[0][1], pd.DataFrame(s_g1, columns = ['Champ'])))

print('Game 2 SKT win rate: {} \n{}\n'.format(clf_lr.predict_proba(X_t_2[-2])[0][1], pd.DataFrame(s_g2, columns = ['Champ'])))

print('Game 3 SKT win rate: {} \n{}\n'.format(clf_lr.predict_proba(X_t_2[-1])[0][1], pd.DataFrame(s_g3, columns = ['Champ'])))
