import os

import pandas as pd



PATH_TO_DATA = '../input/'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')
df_train_features.shape
df_train_features.head()
df_train_targets.head()
X = df_train_features.values

y = df_train_targets['radiant_win'].values
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, 

                                                      random_state=17)
%%time

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_valid)[:, 1]
y_pred
from sklearn.metrics import roc_auc_score



valid_score = roc_auc_score(y_valid, y_pred)

print('Validation ROC-AUC score:', valid_score)
from sklearn.metrics import accuracy_score



valid_accuracy = accuracy_score(y_valid, y_pred > 0.5)

print('Validation accuracy of P>0.5 classifier:', valid_accuracy)
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')



X_test = df_test_features.values

y_test_pred = model.predict_proba(X_test)[:, 1]



df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, 

                                 index=df_test_features.index)
df_submission.head()
import datetime

submission_filename = 'submission_{}.csv'.format(

    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

df_submission.to_csv(submission_filename)

print('Submission saved to {}'.format(submission_filename))
from sklearn.model_selection import ShuffleSplit, KFold

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=17)
from sklearn.model_selection import cross_val_score
%%time



model_rf1 = RandomForestClassifier(n_estimators=100, n_jobs=4,

                                   max_depth=None, random_state=17)



# calcuate ROC-AUC for each split

cv_scores_rf1 = cross_val_score(model_rf1, X, y, cv=cv, scoring='roc_auc')
%%time



model_rf2 = RandomForestClassifier(n_estimators=100, n_jobs=4,

                                   min_samples_leaf=3, random_state=17)



cv_scores_rf2 = cross_val_score(model_rf2, X, y, cv=cv, 

                                scoring='roc_auc', n_jobs=-1)
cv_scores_rf1
cv_scores_rf2
print('Model 1 mean score:', cv_scores_rf1.mean())

print('Model 2 mean score:', cv_scores_rf2.mean())
cv_scores_rf2 > cv_scores_rf1
import json



with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:

    # read the 18-th line

    for i in range(18):

        line = fin.readline()

    

    # read JSON into a Python object 

    match = json.loads(line)
#match
player = match['players'][2]
player['kills'], player['deaths'], player['assists']
player['ability_uses']
%matplotlib inline

from matplotlib import pyplot as plt
for player in match['players']:

    plt.plot(player['times'], player['gold_t'])

    

plt.title('Gold change for all players');
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
for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):

    match_id_hash = match['match_id_hash']

    game_time = match['game_time']

    

    # processing each game

    

    for player in match['players']:

        pass  # processing each player
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

        

        # ... here you can add more features ...

        
# copy the dataframe with features

df_train_features_extended = df_train_features.copy()



# add new features

add_new_features(df_train_features_extended, 

                 os.path.join(PATH_TO_DATA, 

                              'train_matches.jsonl'))
df_train_features_extended.head()
%%time



from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)



cv_scores_base = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

cv_scores_extended = cross_val_score(model, df_train_features_extended.values, y, 

                                     cv=cv, scoring='roc_auc', n_jobs=-1)
print('Base features: mean={} scores={}'.format(cv_scores_base.mean(), 

                                                cv_scores_base))

print('Extended features: mean={} scores={}'.format(cv_scores_extended.mean(), 

                                                    cv_scores_extended))
cv_scores_extended > cv_scores_base
%%time

# Build the same features for the test set

df_test_features_extended = df_test_features.copy()

add_new_features(df_test_features_extended, 

                 os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))
model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)

model.fit(X, y)

df_submission_base = pd.DataFrame(

    {'radiant_win_prob': model.predict_proba(df_test_features.values)[:, 1]}, 

    index=df_test_features.index,

)

df_submission_base.to_csv('submission_base_rf.csv')
model_extended = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)

model_extended.fit(df_train_features_extended.values, y)

df_submission_extended = pd.DataFrame(

    {'radiant_win_prob': model_extended.predict_proba(df_test_features_extended.values)[:, 1]}, 

    index=df_test_features.index,

)

df_submission_extended.to_csv('submission_extended_rf.csv')
# this one will be used as a final submission in this kernel

!cp submission_extended_rf.csv submission.csv
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

            

    return collections.OrderedDict(row)

    

def extract_targets_csv(match, targets):

    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [

        (field, targets[field])

        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']

    ])
%%time



df_new_features = []

df_new_targets = []



for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):

    match_id_hash = match['match_id_hash']

    features = extract_features_csv(match)

    targets = extract_targets_csv(match, match['targets'])

    

    df_new_features.append(features)

    df_new_targets.append(targets)

    
df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')

df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')
df_new_features.head()