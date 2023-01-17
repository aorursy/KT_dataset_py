import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression



PATH_TO_DATA = '../input/'
import collections



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



def sum_features(feature_log):

    sum = 0

    for key in feature_log:

        sum += feature_log[key]

    return sum



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



    # Counting ruined towers for both teams

    radiant_tower_kills = 0

    dire_tower_kills = 0

    for objective in match['objectives']:

        if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':

            if objective['team'] == 2:

                radiant_tower_kills += 1

            if objective['team'] == 3:

                dire_tower_kills += 1



    # Write new features

    row.append(('radiant_tower_kills', radiant_tower_kills))

    row.append(('dire_tower_kills', dire_tower_kills))

    row.append(('diff_tower_kills', radiant_tower_kills - dire_tower_kills))

    

    return collections.OrderedDict(row)

    

def extract_targets_csv(match, targets):

    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [

        (field, targets[field])

        for field in ['radiant_win']

    ])
def extract_inverse_features_csv(match):

    row = [

        ('match_id_hash', match['match_id_hash'][::-1]),

    ]

    

    for field, f in MATCH_FEATURES:

        row.append((field, f(match)))

        

    for slot, player in enumerate(match['players']):

        if slot < 5:

            player_name = 'd%d' % (slot + 1)

        else:

            player_name = 'r%d' % (slot - 4)



        for field in PLAYER_FIELDS:

            column_name = '%s_%s' % (player_name, field)

            row.append((column_name, player[field]))



    # Counting ruined towers for both teams

    radiant_tower_kills = 0

    dire_tower_kills = 0

    for objective in match['objectives']:

        if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':

            if objective['team'] == 3:

                radiant_tower_kills += 1

            if objective['team'] == 2:

                dire_tower_kills += 1



    # Write new features

    row.append(('radiant_tower_kills', radiant_tower_kills))

    row.append(('dire_tower_kills', dire_tower_kills))

    row.append(('diff_tower_kills', radiant_tower_kills - dire_tower_kills))

    

    return collections.OrderedDict(row)



def extract_inverse_targets_csv(match, targets):

    return collections.OrderedDict([('match_id_hash', match['match_id_hash'][::-1])] + [

        (field,  not targets[field])

        for field in ['radiant_win']

    ])
import os

import pandas as pd

import numpy as np





import os



try:

    import ujson as json

except ModuleNotFoundError:

    import json

    print ('Please install ujson to read JSON oblects faster')

    

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
%%time



df_features = []

df_targets = []



df_inverse_features = []

df_inverse_targets = []



for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):

    features = extract_features_csv(match)

    inverse_features = extract_inverse_features_csv(match)

    targets = extract_targets_csv(match, match['targets'])

    inverse_targets = extract_inverse_targets_csv(match, match['targets'])

    

    df_features.append(features)

    df_inverse_features.append(features)

    df_inverse_features.append(inverse_features)

    df_targets.append(targets)

    df_inverse_targets.append(targets)

    df_inverse_targets.append(inverse_targets)
train_df = pd.DataFrame.from_records(df_features).set_index('match_id_hash')

y_train = pd.DataFrame.from_records(df_targets).set_index('match_id_hash')

y_train = y_train['radiant_win'].map({True: 1, False: 0})
y_train.hist()
train_df_inverse = pd.DataFrame.from_records(df_inverse_features).set_index('match_id_hash')

y_train_inverse = pd.DataFrame.from_records(df_inverse_targets).set_index('match_id_hash')

y_train_inverse = y_train_inverse['radiant_win'].map({True: 1, False: 0})
print(train_df.shape)

print(train_df_inverse.shape)
train_df.head()
train_df_inverse.head()
logit_pipe = Pipeline([('scaler', MinMaxScaler(feature_range=(0, 1))),

                       ('logit', LogisticRegression(C=0.5, random_state=17, solver='liblinear'))])



logit_res = cross_val_score(logit_pipe, train_df, y_train, scoring='roc_auc', cv = 5, n_jobs=6)

logit_res.mean()
logit_pipe = Pipeline([('scaler', MinMaxScaler(feature_range=(0, 1))),

                       ('logit', LogisticRegression(C=0.5, random_state=17, solver='liblinear'))])



logit_res = cross_val_score(logit_pipe, train_df_inverse, y_train_inverse, scoring='roc_auc', cv = 5, n_jobs=6)

logit_res.mean()