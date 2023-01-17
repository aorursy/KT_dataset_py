import pandas as pd

import numpy as np

from pathlib import Path

import warnings

warnings.filterwarnings("ignore")

import time

from tqdm import tqdm_notebook

import os

import collections
PATH_TO_DATA = Path('/kaggle/input/mlcourse-dota2-win-prediction')
target = pd.read_csv(PATH_TO_DATA /'train_targets.csv', index_col='match_id_hash')

train_initial = pd.read_csv(PATH_TO_DATA /'train_features.csv', index_col='match_id_hash')

test_initial = pd.read_csv(PATH_TO_DATA /'test_features.csv', index_col='match_id_hash')



target['radiant_win'] = target['radiant_win'].map({False: 0, True: 1})
MATCH_FEATURES = [

    ('game_time', lambda m: m['game_time']),

    ('game_mode', lambda m: m['game_mode']),

    ('lobby_type', lambda m: m['lobby_type']),

    ('objectives_len', lambda m: len(m['objectives'])),

    ('chat_len', lambda m: len(m['chat'])),

]



PLAYER_FIELDS = [

    'hero_id',

    

    'kills',

    'deaths',

    'assists',

    'denies',

    

    'gold',

    'lh',

    'xp',

    'health',

    'max_health',

    'max_mana',

    'level',



    'x',

    'y',

    

    'stuns',

    'creeps_stacked',

    'camps_stacked',

    'rune_pickups',

    'firstblood_claimed',

    'teamfight_participation',

    'towers_killed',

    'roshans_killed',

    'obs_placed',

    'sen_placed',



]





def extract_features_csv(match):

    row = [

        ('match_id_hash', match['match_id_hash']),

    ]

    

    for field, f in MATCH_FEATURES:

        row.append((field, f(match)))

        

    for slot, player in enumerate(match['players']):

        if slot < 5:

            player_name = 'r%d' % (slot + 1)

        else:

            player_name = 'd%d' % (slot - 4)



        for field in PLAYER_FIELDS:

            column_name = '%s_%s' % (player_name, field)

            row.append((column_name, player[field]))

        

        for field in ['ability_upgrades']:

            column_name = '%s_%s' % (player_name, field)

            row.append((column_name, len(player[field])))

        

        for field in ['damage', 'damage_taken']:

            column_name = '%s_%s' % (player_name, field)

            row.append((column_name, sum(list(player[field].values()))))

        

        row.append( (f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory'])) ) ) # return the list of items

            

    return collections.OrderedDict(row)
try:

    import ujson as json

except ModuleNotFoundError:

    import json

    print ('Please install ujson to read JSON objects faster')

    

try:

    from tqdm import tqdm_notebook

except ModuleNotFoundError:

    tqdm_notebook = lambda x: x

    print ('Please install tqdm to track progress with Python loops')



def read_matches(matches_file):

    

    MATCHES_COUNT = {

        'test_matches.jsonl': 10000,

        'train_matches.jsonl': 39675,

    }

    _, filename = os.path.split(matches_file)

    total_matches = MATCHES_COUNT.get(filename)

    

    with open(matches_file) as fin:

        for line in tqdm_notebook(fin, total=total_matches):

            yield json.loads(line)
df_new_features = []

df_new_features_test = []
for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):

    match_id_hash = match['match_id_hash']

    features = extract_features_csv(match)

    df_new_features.append(features)
for match in read_matches(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')):

    match_id_hash = match['match_id_hash']

    features = extract_features_csv(match)

    df_new_features_test.append(features)
train_extracted = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')

test_extracted = pd.DataFrame.from_records(df_new_features_test).set_index('match_id_hash')
def add_new_features(df_features, matches_file):

    

    # Process raw data and add new features

    for match in read_matches(matches_file):

        match_id_hash = match['match_id_hash']



        # Counting ruined towers for both teams

        radiant_tower_kills = 0

        dire_tower_kills = 0

        radiant_baracks_kills = 0

        dire_baracks_kills = 0

        radiant_aegis = 0

        dire_aegis = 0

        for objective in match['objectives']:

            if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL': # feature presented in yorko's kernel as well, usefulness of this feature is obvious

                if objective['team'] == 2:

                    radiant_tower_kills += 1

                if objective['team'] == 3:

                    dire_tower_kills += 1

            if objective['type'] == 'CHAT_MESSAGE_TOWER_DENY': # tower deny is some kind of extension of tower_kill feature (pay attention, if the team denies their tower - the enemy gets +1 tower kill)

                if objective['player_slot'] in [128, 129, 130, 131, 132]:

                    radiant_tower_kills += 1

                if objective['player_slot'] in [0,1,2,3,4]:

                    dire_tower_kills += 1

            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL': # barracks situated nearly opposite team's base, which means that barrack kills may be potentially very strong feature 

                                                                    #(be attentive, this time keys are in string format)

                if objective['key'] in ['1','2','4','8','16','32']:

                    radiant_baracks_kills += 1

                if objective['key'] in ['64','128','256','512','1024','2048']:

                    dire_baracks_kills += 1

            if objective['type'] == 'CHAT_MESSAGE_AEGIS': # aegis grants an extra life, roshan drops it each time he dies (so may be correlated with roshan feature)

                if objective['player_slot'] in [0,1,2,3,4]:

                    radiant_aegis += 1

                if objective['player_slot'] in [128, 129, 130, 131, 132]: 

                    dire_aegis += 1

                

        

        r_chat = 0

        d_chat = 0

        r_memb_chat = []

        d_memb_chat = []

        for chat in match['chat']:

            if chat['player_slot'] in [0,1,2,3,4]:

                r_chat += 1

                r_memb_chat.append(chat['player_slot'])

            if chat['player_slot'] in [128, 129, 130, 131, 132]: 

                d_chat += 1

                d_memb_chat.append(chat['player_slot'])

        

        # Write new features

        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills

        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills

        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills

        

        df_features.loc[match_id_hash, 'radiant_baracks_kills'] = radiant_baracks_kills

        df_features.loc[match_id_hash, 'dire_baracks_kills'] = dire_baracks_kills

        df_features.loc[match_id_hash, 'diff_baracks_kills'] = radiant_baracks_kills - dire_baracks_kills

        

        df_features.loc[match_id_hash, 'radiant_aegis'] = radiant_aegis

        df_features.loc[match_id_hash, 'dire_aegis'] = dire_aegis

        df_features.loc[match_id_hash, 'diff_aegis'] = radiant_aegis - dire_aegis

        

        df_features.loc[match_id_hash, 'radiant_chat_len'] = r_chat

        df_features.loc[match_id_hash, 'dire_chat_len'] = d_chat

        df_features.loc[match_id_hash, 'diff_chat_len'] = r_chat - d_chat # return chat length for each team, not the common one as in initial dataset

        

        df_features.loc[match_id_hash, 'radiant_chat_memb'] = len(np.unique(r_memb_chat))

        df_features.loc[match_id_hash, 'dire_chat_memb'] = len(np.unique(d_memb_chat))

        df_features.loc[match_id_hash, 'diff_chat_memb'] = len(np.unique(r_memb_chat)) - len(np.unique(d_memb_chat)) # return the number of team members, who participate in chatting
add_new_features(train_extracted, 

                 os.path.join(PATH_TO_DATA, 

                              'train_matches.jsonl'))

add_new_features(test_extracted, 

                 os.path.join(PATH_TO_DATA, 

                              'test_matches.jsonl'))
# function creates damage features as well as difference between damage given and taken for each player

def add_damage(df):

    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)]

        for player in players:

            df[f'{player}_dam_diff'] = df[f'{player}_damage'] - df[f'{player}_damage_taken']

    df.drop(df.filter(like = 'damage').columns, axis = 1, inplace = True)

    return df
add_damage(train_extracted)

add_damage(test_extracted);
[x for x in list(train_extracted.columns) if x not in list(train_initial.columns)]
train_extracted[[x for x in list(train_extracted.columns) if x not in list(train_initial.columns)]].head()
train_extracted.to_pickle('train_extracted.pkl')

test_extracted.to_pickle('test_extracted.pkl')

target.to_pickle('target.pkl')
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import ShuffleSplit, KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=17)

model_rf = RandomForestClassifier(n_estimators=100, n_jobs=4,

                                   min_samples_leaf=3, random_state=17)
X_initial = train_initial.values

X_extracted = train_extracted[[x for x in list(train_extracted.columns) if x not in list(train_extracted.filter(like = 'item').columns)]].values

y = target['radiant_win'].values
%%time

cv_scores_rf_initial = cross_val_score(model_rf, X_initial, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)
%%time

cv_scores_rf_extracted = cross_val_score(model_rf, X_extracted, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)
# That's perfectly results of yorko's initial RF model

cv_scores_rf_initial
cv_scores_rf_extracted
cv_scores_rf_extracted > cv_scores_rf_initial