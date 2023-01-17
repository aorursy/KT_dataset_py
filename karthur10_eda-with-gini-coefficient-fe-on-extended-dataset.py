import pandas as pd

import numpy as np

from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

from sklearn import metrics as sm
PATH_TO_DATA = Path('/kaggle/input/data-extraction-from-json-additional-features')
train_df = pd.read_pickle(PATH_TO_DATA/'train_extracted.pkl')

test_df = pd.read_pickle(PATH_TO_DATA/'test_extracted.pkl')

target = pd.read_pickle(PATH_TO_DATA/'target.pkl')
train_df.head(3)
target.head(3)
f'Train set\'s shape is {train_df.shape}, of the test set is {test_df.shape} and targer set\'s shape is {target.shape}.'
# There are no missed values for the train set

for i in train_df.columns:

    if train_df[i].isnull().sum() > 0:

        print(i, train_df[i].isnull().sum())
# So as for the test set

for i in test_df.columns:

    if test_df[i].isnull().sum() > 0:

        print(i, test_df[i].isnull().sum())
full_df = pd.concat([train_df, test_df], sort=False)

full_df.shape
if all(train_df.columns == test_df.columns):

    print('Train and test features are identical')



if len(full_df.index.unique()) == len(full_df.index):

    print('There is no repeating games in the train and test datasets.')
def gini(fpr, tpr):

    """

    Function calculates Gini coefficient.

    fpr - the vector of class labels;

    tpr - the vector of feature(s) values.

    """

    return -(2 * sm.roc_auc_score(fpr, tpr) - 1)
gini_df = {}

for i in [x for x in list(train_df.columns) if x not in list(train_df.filter(like = 'item').columns)]:

    gini_df[i] = gini(target['radiant_win'].values, train_df[i])

gini_df = pd.DataFrame.from_dict(gini_df, orient = 'index',columns = ['gini'])

gini_df['gini_abs'] = abs(gini_df['gini'])

gini_df = gini_df.sort_values('gini_abs', ascending = False)
# top 10, baracks_kills and tower_kills are promising

gini_df.head(10)
# 10 loosers

gini_df.tail(10).index
# Let's plot our data.

import seaborn as sns

from matplotlib import pyplot as plt
#Just a slight disbalance

sns.countplot(target['radiant_win'])

plt.title('Result distribution')

plt.show();
# Seems pretty the same for both samples.

fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

sns.distplot(train_df['game_time'], ax = ax[0][0])

sns.distplot(test_df['game_time'], ax = ax[0][1])

ax[0][0].set_title('Train')

ax[0][1].set_title('Test')

plt.show();
sns.distplot(target.duration)

plt.title('Duration')

plt.show();
train_avg_time = round(train_df['game_time'].mean(),2)

test_avg_time = round(test_df['game_time'].mean(),2)

train_std_time = round(train_df['game_time'].std(),2)

test_std_time = round(test_df['game_time'].std(),2)

print(f'Average game_time for train - {train_avg_time}, test - {test_avg_time}.')

print(f'Standard deviation for train = {train_std_time}, for test = {test_std_time}.')
fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

sns.distplot(train_df[target['radiant_win'] == 1]['game_time'], hist = False, label = 'R_WIN',ax = ax[0][0])

sns.distplot(train_df[target['radiant_win'] == 0]['game_time'], hist = False, label = 'D_WIN', ax = ax[0][0])

sns.distplot(target[target['radiant_win'] == 1]['duration'], hist = False, label = 'R_WIN', ax = ax[0][1])

sns.distplot(target[target['radiant_win'] == 0]['duration'], hist = False, label = 'D_WIN', ax = ax[0][1])

plt.legend()

plt.show();
# Quite a low value of index

g_game_time = gini(target['radiant_win'].values, train_df['game_time'])

g_duration = gini(target['radiant_win'].values, target['duration'])

print(f'Gini for game_time  = {g_game_time}')

print(f'Gini for duration  = {g_duration}') 
# What about game_mode?

fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

a = sns.countplot(train_df['game_mode'], ax = ax[0][0])

for p in a.patches:

    a.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),\

                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

b = sns.countplot(test_df['game_mode'], ax = ax[0][1])

for p in b.patches:

    b.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),\

                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

ax[0][0].set_title('Train')

ax[0][1].set_title('Test')

plt.show();
fig = plt.figure(figsize=(8, 5)) 

sns.countplot(train_df['game_mode'], hue = target['radiant_win'])

plt.show();
pd.crosstab(train_df['game_mode'], target['radiant_win'], normalize = 'index').sort_values(1)
pd.crosstab(train_df['game_mode'], columns = target['radiant_win'], values = target['duration'], aggfunc = 'mean')
pd.crosstab(train_df['game_mode'], columns = target['radiant_win'], values = target['game_time'], aggfunc = 'mean')
train_df['lobby_type'].value_counts(normalize = True)
test_df['lobby_type'].value_counts(normalize = True)
# Radiant_win rate by lobby_type distribution in game_time

fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

sns.distplot(train_df[train_df['lobby_type'] == 0]['game_time'], hist = False, label = '0', ax = ax[0][0])

sns.distplot(train_df[train_df['lobby_type'] == 7]['game_time'], hist = False, label = '7', ax = ax[0][0])

sns.distplot(test_df[test_df['lobby_type'] == 0]['game_time'], hist = False, label = '0', ax = ax[0][1])

sns.distplot(test_df[test_df['lobby_type'] == 7]['game_time'], hist = False, label = '7', ax = ax[0][1])

ax[0][0].set_title('Train')

ax[0][1].set_title('Test')

plt.legend()

plt.show();
# Objective_len

fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

a = sns.countplot(train_df['objectives_len'], ax = ax[0][0])

a.set_xticklabels(a.get_xticklabels(),rotation=90, horizontalalignment='right')

b = sns.countplot(test_df['objectives_len'], ax = ax[0][1])

b.set_xticklabels(b.get_xticklabels(),rotation=90, horizontalalignment='right')

ax[0][0].set_title('Train')

ax[0][1].set_title('Test')

plt.show();
train_df[train_df.iloc[:,:5].columns].corr()
fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)

a = sns.scatterplot(train_df['chat_len'], train_df['chat_len'], ax = ax[0][0])

b = sns.scatterplot(test_df['chat_len'], test_df['chat_len'], ax = ax[0][1])

ax[0][0].set_title('Train')

ax[0][1].set_title('Test')

plt.show();
train_df.filter(like = 'chat').columns
g_chat_len = gini(target['radiant_win'].values, train_df['chat_len'].values)

g_radiant_chat_len = gini(target['radiant_win'].values, train_df['radiant_chat_len'].values)

g_dire_chat_len = gini(target['radiant_win'].values, train_df['dire_chat_len'].values)

g_diff_chat_len = gini(target['radiant_win'].values, train_df['diff_chat_len'].values)

g_radiant_chat_memb = gini(target['radiant_win'].values, train_df['radiant_chat_memb'].values)

g_dire_chat_memb = gini(target['radiant_win'].values, train_df['dire_chat_memb'].values)

g_diff_chat_memb = gini(target['radiant_win'].values, train_df['diff_chat_memb'].values)

print(f'Gini for chat_len = {g_chat_len}, radiant_chat_len = {g_radiant_chat_len}, dire_chat_len = {g_dire_chat_len}, diff_chat_len = {g_diff_chat_len}')

print(f'radiant_chat_memb = {g_radiant_chat_memb}, dire_chat_memb = {g_dire_chat_memb}, diff_chat_memb = {g_diff_chat_memb}')
train_size = train_df.shape[0]

hero_columns = [c for c in full_df.columns if '_hero_' in c]

train_hero_id = train_df[hero_columns]

train_hero_id.head(3)
for team in 'r', 'd':

    players = [f'{team}{i}' for i in range(1, 6)]

    hero_columns = [f'{player}_hero_id' for player in players]

    d = pd.get_dummies(full_df[hero_columns[0]])

    for c in hero_columns[1:]:

        d += pd.get_dummies(full_df[c])

    full_df = pd.concat([full_df, d.add_prefix(f'{team}_hero_')], axis=1)

    full_df.drop(columns=hero_columns, inplace=True)

    

train_df = full_df.iloc[:train_size, :]

test_df = full_df.iloc[train_size:, :]
if all(train_df.filter(like = 'hero').columns == test_df.filter(like = 'hero').columns):

    print('hero_ids in the train sample are the same as in the test sample.')
train_df[train_df.filter(like = 'hero').columns].corrwith(target.radiant_win).abs().sort_values(ascending=False).head(12)
train_df[train_df.filter(like = 'hero').columns].corrwith(target.radiant_win).abs().sort_values(ascending=False).tail(12)
heroes = pd.DataFrame(train_df[train_df.filter(like = 'hero').columns].sum().sort_values(ascending = False)

    , columns = ['Train']).merge(pd.DataFrame(test_df[test_df.filter(like = 'hero').columns].sum()

    , columns = ['Test']), left_index = True, right_index = True)

heroes['train_occ'] = round(heroes['Train']/train_df.shape[0]*100,2)

heroes['test_occ'] = round(heroes['Test']/test_df.shape[0]*100,2)

heroes.head(12)
# Kinda good news

heroes[['train_occ', 'test_occ']].corr()
# What about our most succesful hero_ids?

heroes.loc[['d_hero_32','r_hero_32','r_hero_22','r_hero_19',

      'd_hero_22','d_hero_19','d_hero_92','d_hero_91',

      'd_hero_73','r_hero_92','r_hero_91']]
train_df.filter(like = 'r1').columns
#let's make our life simplier

def combine_numeric_features (df, feature_suffixes):

    for feat_suff in feature_suffixes:

        for team in 'r', 'd':

            players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...

            player_col_names = [f'{player}_{feat_suff}' for player in players] # e.g. r1_gold, r2_gold

            

            df[f'{team}_{feat_suff}_max'] = df[player_col_names].max(axis=1) # e.g. r_gold_max

            df[f'{team}_{feat_suff}_mean'] = df[player_col_names].mean(axis=1) # e.g. r_gold_mean

            df[f'{team}_{feat_suff}_min'] = df[player_col_names].min(axis=1) # e.g. r_gold_min

            df[f'{team}_{feat_suff}_sum'] = df[player_col_names].sum(axis=1)

            df[f'{team}_{feat_suff}_std'] = df[player_col_names].std(axis=1)

            

            df.drop(columns=player_col_names, inplace=True) # remove raw features from the dataset

            

    return df
numeric_features = ['kills', 'deaths', 'assists', 'denies', 'gold', 'xp', 'health', 

                    'max_health', 'max_mana', 'level', 'towers_killed', 'stuns', 'creeps_stacked', 

                    'camps_stacked', 'lh', 'rune_pickups', 'firstblood_claimed', 'teamfight_participation', 

                    'roshans_killed', 'obs_placed', 'sen_placed', 'dam_diff', 'ability_upgrades']
train_df = combine_numeric_features(train_df, numeric_features)

test_df = combine_numeric_features(test_df, numeric_features)
train_df.head(3)
# Creating vectors of x and y

x_values = []

y_values = []

for team in 'r','d':

    players = [f'{team}{i}' for i in range(1, 6)]

    for i in players:

        x_values += list(train_df[f'{i}_x'])

        y_values += list(train_df[f'{i}_y'])

coord_df = pd.DataFrame(x_values, columns = ['x'])

coord_df['y'] = y_values

coord_df['radiant_win'] = list(target['radiant_win'])*10

coord = pd.pivot_table(data = coord_df, index = 'y', columns = 'x', values = 'radiant_win', aggfunc = 'mean').fillna(0)
fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(coord.sort_index(ascending = False), ax = ax);
def make_coordinate_features(df):

    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...

        for player in players:

            df[f'{player}_distance'] = np.sqrt(df[f'{player}_x']**2 + df[f'{player}_y']**2)

            df.drop(columns=[f'{player}_x', f'{player}_y'], inplace=True)

    return df
train_df = make_coordinate_features(train_df)

test_df = make_coordinate_features(test_df)
train_df = combine_numeric_features(train_df, ['distance'])

test_df = combine_numeric_features(test_df, ['distance'])
# As expected baracks_kills feature is quite strong

baracks = pd.crosstab(train_df['diff_baracks_kills'],target['radiant_win'], normalize = 'index')

sns.lineplot(y = baracks.index, x=baracks[1]);
# But baracks are killed mostly in the end of the game, so often diff_baracks_kills = 0

train_df['diff_baracks_kills'].value_counts()
towers = pd.crosstab(train_df['diff_tower_kills'],target['radiant_win'], normalize = 'index')

sns.lineplot(y = towers.index, x=towers[1]);
# A bit better with towers

train_df['diff_tower_kills'].value_counts()
aegis = pd.crosstab(train_df['diff_aegis'],target['radiant_win'], normalize = 'index')

sns.lineplot(y = aegis.index, x=aegis[1]);
def add_items_dummies(df_train, df_test):

    

    full_df = pd.concat([df_train, df_test], sort=False)

    train_size = df_train.shape[0]



    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)]

        item_columns = [f'{player}_items' for player in players]



        d = pd.get_dummies(full_df[item_columns[0]].apply(pd.Series).stack()).sum(level=0, axis=0)

        dindexes = d.index.values



        for c in item_columns[1:]:

            d = d.add(pd.get_dummies(full_df[c].apply(pd.Series).stack()).sum(level=0, axis=0), fill_value=0)

            d = d.ix[dindexes]



        full_df = pd.concat([full_df, d.add_prefix(f'{team}_item_')], axis=1, sort=False)

        full_df.drop(columns=item_columns, inplace=True)



    df_train = full_df.iloc[:train_size, :]

    df_test = full_df.iloc[train_size:, :]



    return df_train, df_test
def drop_consumble_items(df_train, df_test):

    

    full_df = pd.concat([df_train, df_test], sort=False)

    train_size = df_train.shape[0]



    for team in 'r', 'd':

        consumble_columns = ['tango', 'tpscroll', 

                             'bottle', 'flask',

                            'enchanted_mango', 'clarity',

                            'faerie_fire', 'ward_observer',

                            'ward_sentry']

        

        starts_with = f'{team}_item_'

        consumble_columns = [starts_with + column for column in consumble_columns]

        full_df.drop(columns=consumble_columns, inplace=True)



    df_train = full_df.iloc[:train_size, :]

    df_test = full_df.iloc[train_size:, :]



    return df_train, df_test
train_df, test_df = add_items_dummies(train_df, test_df)

train_df, test_df = drop_consumble_items(train_df, test_df)
train_df.columns
train_df[train_df.filter(like = 'item').columns].corrwith(target.radiant_win).abs().sort_values(ascending=False).head(20)
gini_df = {}

for i in [x for x in list(train_df.columns) if x not in list(train_df.filter(like = 'item').columns)]:

    gini_df[i] = gini(target['radiant_win'].values, train_df[i])

gini_df = pd.DataFrame.from_dict(gini_df, orient = 'index',columns = ['gini'])

gini_df['gini_abs'] = abs(gini_df['gini'])

gini_df = gini_df.sort_values('gini_abs', ascending = False)
gini_df.head(40)