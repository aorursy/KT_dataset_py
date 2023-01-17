import os

import json



import pandas as pd



PATH_TO_DATA = '../input/'

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), index_col='match_id_hash')



df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), index_col='match_id_hash')
df_train_features.head(2)
json_list = []

number_of_rows = 50 # Number of readed json rows



with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:

    for i in range(number_of_rows):

        line = fin.readline()

        json_list.append(json.loads(line))
json_list[0].keys()
json_list[0]['chat']
json_list[1]['objectives']

print(json.dumps(json_list[1]['teamfights'][1], indent=4, sort_keys=True))
print(json.dumps(json_list[1]['players'][1], indent=4, sort_keys=True))
y = df_train_targets['radiant_win'].values
try:

    from tqdm import tqdm_notebook

except ModuleNotFoundError:

    tqdm_notebook = lambda x: x

    print ('Please install tqdm to track progress with Python loops')
#a helper function, we will use it in next cell

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
def add_new_features(df_features, matches_file):

    

    # Process raw data and add new features

    for match in read_matches(matches_file):

        match_id_hash = match['match_id_hash']



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

        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills

        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills

        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills

        

        
%%time

# copy the dataframe with features

df_train_features_extended = df_train_features.copy()



# add new features

add_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))

from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier

lgb_classifier = LGBMClassifier(n_estimators=200)
cv_default = cross_val_score(lgb_classifier, df_train_features_extended, y, cv=5)

cv_default.mean()
def add_new_features(df_features, matches_file):

    

    # Process raw data and add new features

    for match in read_matches(matches_file):

        match_id_hash = match['match_id_hash']



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

        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills

        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills

        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills

        

        # Total damage

        total_damage = 0

        for i in range(1, 6):

            for j in match['players'][i-1]['damage']:

                # Take damage only to hero(not for creeps)

                if j.startswith('npc_dota_hero'):

                    total_damage += match['players'][i-1]['damage'][j]

        df_features.loc[match_id_hash, 'r_damage'] = total_damage

        total_damage = 0

        for i in range(6, 11):

            for j in match['players'][i-1]['damage']:

                if j.startswith('npc_dota_hero'):

                    total_damage += match['players'][i-1]['damage'][j]

        df_features.loc[match_id_hash, 'd_damage'] = total_damage



        df_features.loc[match_id_hash, 'diff_damage'] = df_features.loc[match_id_hash, 'r_damage'] - df_features.loc[match_id_hash, 'd_damage'] 



      
%%time

# copy the dataframe with features

df_train_features_extended = df_train_features.copy()



# add new features

add_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))

cv_extended = cross_val_score(lgb_classifier, df_train_features_extended, y, cv=5)

cv_extended.mean()