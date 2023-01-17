from scipy.sparse import hstack

import time

import os

import lightgbm as lgb

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, StratifiedKFold, KFold, train_test_split, ShuffleSplit

from sklearn.preprocessing import StandardScaler

#import ujson as json

from tqdm import tqdm_notebook

import collections

from catboost import CatBoostClassifier, Pool, cv
PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction/'

train_df = pd.read_csv(PATH_TO_DATA + 'train_features.csv')

train_df.head()
MATCH_FEATURES = [

    ('game_time', lambda m: m['game_time']),

    ('game_mode', lambda m: m['game_mode']),

    ('lobby_type', lambda m: m['lobby_type']),

    ('objectives_len', lambda m: len(m['objectives'])),

    ('chat_len', lambda m: len(m['chat'])),

]



PLAYER_FIELDS = [

    'hero_id',

    'gold',

    'x',

    'y'

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

        

        for field in ['damage']:

            column_name = '%s_%s' % (player_name, field)

            row.append((column_name, sum(list(player[field].values()))))



            

    return collections.OrderedDict(row)

    

def extract_targets_csv(match, targets):

    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [

        (field, targets[field])

        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']

    ])



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

            

train_features_from_json = []

targets_from_json = []



for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):

    match_id_hash = match['match_id_hash']

    features = extract_features_csv(match)

    target = extract_targets_csv(match, match['targets'])

    

    train_features_from_json.append(features)

    targets_from_json.append(target)

    

test_features_from_json = []



for match in read_matches(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')):

    match_id_hash = match['match_id_hash']

    features = extract_features_csv(match)

    test_features_from_json.append(features)
train = pd.DataFrame.from_records(train_features_from_json).set_index('match_id_hash')

test = pd.DataFrame.from_records(test_features_from_json).set_index('match_id_hash')

targets = pd.DataFrame.from_records(targets_from_json).set_index('match_id_hash')

y = targets['radiant_win']



train.head()
test.head()
[c for c in train.columns if 'r1_' in c]
rf = RandomForestClassifier(n_estimators=100, 

                            n_jobs=4, 

                            min_samples_leaf=3,

                            random_state=17)



n_fold = 5

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=17)

cv_scores_rf_initial = cross_val_score(rf, train, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)
cv_scores_rf_initial
cv_scores_rf_initial.mean(), cv_scores_rf_initial.std()


def add_new_features(df_features, matches_file):

    

    # Process raw data and add new features

    for match in read_matches(matches_file):

        match_id_hash = match['match_id_hash']



        radiant_baracks_kills = 0

        dire_baracks_kills = 0



        for objective in match['objectives']:



            if objective['type'] == 'CHAT_MESSAGE_BARRACKS_KILL':

                if objective['key'] in ['1','2','4','8','16','32']:

                    radiant_baracks_kills += 1

                if objective['key'] in ['64','128','256','512','1024','2048']:

                    dire_baracks_kills += 1



        df_features.loc[match_id_hash, 'radiant_baracks_kills'] = radiant_baracks_kills

        df_features.loc[match_id_hash, 'dire_baracks_kills'] = dire_baracks_kills

        df_features.loc[match_id_hash, 'diff_baracks_kills'] = radiant_baracks_kills - dire_baracks_kills

new_train = pd.DataFrame(index = train.index)

new_test = pd.DataFrame(index = test.index)
add_new_features(new_train, 

                 os.path.join(PATH_TO_DATA, 

                              'train_matches.jsonl'))

add_new_features(new_test, 

                 os.path.join(PATH_TO_DATA, 

                              'test_matches.jsonl'))
new_train['radiant_baracks_kills'].describe()
train1 = pd.concat([train, new_train], axis = 1)
cv_scores_rf1 = cross_val_score(rf, train1, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)
cv_scores_rf1.mean(), cv_scores_rf1.std()
cv_scores_rf1 > cv_scores_rf_initial
def modify_df_with_feature(dataframe, feature_name):

    epsilon = 0.00000000000001             # Let's avoid zero devision situation

    dataframe = dataframe.copy()

    r_feature_columns = [col for col in dataframe if col.endswith(feature_name)][:5]

    d_feature_columns = [col for col in dataframe if col.endswith(feature_name)][5:]

    dataframe['r_' + feature_name + '_total'] = dataframe[r_feature_columns].sum(axis = 1)

    dataframe['d_' + feature_name + '_total'] = dataframe[d_feature_columns].sum(axis = 1)

    dataframe[feature_name + '_prop'] = dataframe['r_' + feature_name + '_total'] / (dataframe['d_' + feature_name + '_total'] + epsilon)   # <- approved



    return dataframe
train2 = modify_df_with_feature(train1, 'gold')
train2.head()
cv_scores_rf2 = cross_val_score(rf, train2, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)
cv_scores_rf2.mean(), cv_scores_rf2.std()
cv_scores_rf2 > cv_scores_rf1
def add_hero_names(df):

    hero_columns = [c for c in df.columns if '_hero_' in c]

    names_df = df[hero_columns]

    names_df = names_df.astype(str)

    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)]

        hero_columns = [f'{player}_hero_id' for player in players]

        d = pd.get_dummies(names_df[hero_columns[0]])

        for c in hero_columns[1:]:

            d += pd.get_dummies(names_df[c])

        names_df = pd.concat([names_df, d.add_prefix(f'{team}_hero_')], axis=1)

        names_df.drop(columns=hero_columns, inplace=True)

    df = pd.concat([df, names_df], axis = 1)

    return df
train3 = add_hero_names(train2)
cv_scores_rf3 = cross_val_score(rf, train3, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)

cv_scores_rf3.mean(), cv_scores_rf3.std()
cv_scores_rf3 > cv_scores_rf2
train.head()
train_up = train.reset_index()

train_up['match_id_hash'] = train_up['match_id_hash'] + '_up'

train_up.set_index('match_id_hash', inplace=True)
features = ['hero_id',

            'gold', 

            'damage',

           ]
for t1 in ['r', 'd']:

    for player in range(1,6):

        for feature in features:

            if t1 == 'r':

                t2 = 'd'

            else:

                t2 = 'r'

            col1 = t1 + str(player) + '_' + feature

            col2 = t2 + str(player) + '_' + feature

            train_up[col1] = train[col2].values

            

        x_col1 = t1 + str(player) + '_x'

        x_col2 = t2 + str(player) + '_x'

        train_up[x_col1] = 186 - (train[x_col2].values - 68)

        y_col1 = t1 + str(player) + '_y'

        y_col2 = t2 + str(player) + '_y'

        train_up[y_col1] = 186 - (train[y_col2].values - 68)
train_up.head()
emb_train = pd.concat([train, train_up])

emb_train.shape
targets_up = targets.reset_index()

targets_up['match_id_hash'] = targets_up['match_id_hash'] + '_up'

targets_up.set_index('match_id_hash', inplace=True)

targets_up['radiant_win'] = targets_up['radiant_win'].map({False: True, True: False})



targets = pd.concat([targets, targets_up])



targets.shape
new_train_up = new_train.reset_index()

new_train_up['match_id_hash'] = new_train_up['match_id_hash'] + '_up'

new_train_up.set_index('match_id_hash', inplace=True)
for feat in ['_baracks_kills']:

    for t1 in ['radiant', 'dire']:

        if t1 == 'radiant':

            t2 = 'dire'

        else:

            t2 = 'radiant'

        col1 = t1 + feat

        col2 = t2 + feat

        new_train_up[col1] = new_train[col2].values



    new_train_up['diff' + feat] = -new_train['diff' + feat].values
emb_new_train = pd.concat([new_train, new_train_up])

emb_new_train.shape
train_emb = pd.concat([emb_train, emb_new_train], axis = 1)

train_emb_gold = modify_df_with_feature(train_emb, 'gold')

train_emb_gold_heroes = add_hero_names(train_emb_gold)

train_emb_gold_heroes.shape

y = targets['radiant_win']
cv_scores_rf_final = cross_val_score(rf, train_emb_gold_heroes, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)

cv_scores_rf_final.mean(), cv_scores_rf_final.std()
cv_scores_rf_final > cv_scores_rf3