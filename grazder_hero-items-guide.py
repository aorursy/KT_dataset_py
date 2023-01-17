import os

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from itertools import combinations

import seaborn as sns

import lightgbm as lgb



import warnings

warnings.filterwarnings("ignore")
%%time

PATH_TO_DATA = '../input'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')



y = df_train_targets['radiant_win'].values
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
json_list = [] #store data that are read

number_of_rows = 50 #how many lines to read 



#reading data from .jsonl file

with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:

    for i in range(number_of_rows):

        line = fin.readline()

        json_list.append(json.loads(line))

        

#how many matches to read. For example I took 1

for i in range(1, 2):

  for j in range(1, 2):#there is 5 players in each team. But I want to look on only one player.

    print(json.dumps(json_list[i]['players'][j], indent=4, sort_keys=True))
for i in range(1, 5): #now we will look at 4 matches

  for j in range(1, 5):#and now will take 5 players

    print(json.dumps(list(map(lambda x: x['id'][5:], json_list[i]['players'][j]['hero_inventory'])), indent=4, sort_keys=True))
import collections





def extract_features_csv(match):

    

    row = [

        ('match_id_hash', match['match_id_hash']),

    ]



    for slot, player in enumerate(match['players']):

        if slot < 5:

            player_name = 'r%d' % (slot + 1)

        else:

            player_name = 'd%d' % (slot - 4)



        row.append( (f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory'])) ) )

        #here u can extract other data



    return collections.OrderedDict(row)



    

def extract_targets_csv(match, targets):

    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [

        (field, targets[field])

        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']

    ])
def create_features_from_jsonl(matches_file):

  

    df_new_features = []



    # Process raw data and add new features

    for match in read_matches(matches_file):

        match_id_hash = match['match_id_hash']

        features = extract_features_csv(match)



        df_new_features.append(features)



    df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')

    return df_new_features
%%time

train_df = create_features_from_jsonl(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')).fillna(0)

test_df = create_features_from_jsonl(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')).fillna(0)
#Let's look at extracted item's data

train_df['r1_items'].head()
import pickle as pkl



#Better to save extracted data in files, because extracting takes time...

train_df.to_pickle('df_train.pkl')

test_df.to_pickle('df_test.pkl')
def add_items_dummies(train_df, test_df):

    

    full_df = pd.concat([train_df, test_df], sort=False)

    train_size = train_df.shape[0]



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



    train_df = full_df.iloc[:train_size, :]

    test_df = full_df.iloc[train_size:, :]



    return train_df, test_df
def drop_consumble_items(train_df, test_df):

    

    full_df = pd.concat([train_df, test_df], sort=False)

    train_size = train_df.shape[0]



    for team in 'r', 'd':

        consumble_columns = ['tango', 'tpscroll', 

                             'bottle', 'flask',

                            'enchanted_mango', 'clarity',

                            'faerie_fire', 'ward_observer',

                            'ward_sentry']

        

        starts_with = f'{team}_item_'

        consumble_columns = [starts_with + column for column in consumble_columns]

        full_df.drop(columns=consumble_columns, inplace=True)



    train_df = full_df.iloc[:train_size, :]

    test_df = full_df.iloc[train_size:, :]



    return train_df, test_df
%%time

new_train = pd.read_pickle('df_train.pkl')

new_test = pd.read_pickle('df_test.pkl')



new_train, new_test = add_items_dummies(new_train, new_test)

new_train, new_test = drop_consumble_items(new_train, new_test)



target = pd.DataFrame(y)
new_train.shape, target.shape, new_test.shape
# Features variable to look at features importance in the end

features = new_train.columns
param = {

        'bagging_freq': 5,  #handling overfitting

        'bagging_fraction': 0.5,  #handling overfitting - adding some noise

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.05, #handling overfitting

        'learning_rate': 0.01,  #the changes between one auc and a better one gets really small thus a small learning rate performs better

        'max_depth': -1,  

        'metric':'auc',

        'min_data_in_leaf': 50,

        'min_sum_hessian_in_leaf': 10.0,

        'num_leaves': 10,

        'num_threads': 5,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1

    }
%%time

#divide training data into train and validaton folds

folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=17)



#placeholder for out-of-fold, i.e. validation scores

oof = np.zeros(len(new_train))



#for predictions

predictions = np.zeros(len(new_test))



#and for feature importance

feature_importance_df = pd.DataFrame()



#RUN THE LOOP OVER FOLDS

for fold_, (trn_idx, val_idx) in enumerate(folds.split(new_train.values, target.values)):

    

    X_train, y_train = new_train.iloc[trn_idx], target.iloc[trn_idx]

    X_valid, y_valid = new_train.iloc[val_idx], target.iloc[val_idx]

    

    print("Computing Fold {}".format(fold_))

    trn_data = lgb.Dataset(X_train, label = y_train)

    val_data = lgb.Dataset(X_valid, label = y_valid)



    

    num_round = 5000 

    verbose=1000 

    stop=500 

    

    #TRAIN THE MODEL

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=verbose, early_stopping_rounds = stop)

    

    #CALCULATE PREDICTION FOR VALIDATION SET

    oof[val_idx] = clf.predict(new_train.iloc[val_idx], num_iteration=clf.best_iteration)

    

    #FEATURE IMPORTANCE

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    #CALCULATE PREDICTIONS FOR TEST DATA, using best_iteration on the fold

    predictions += clf.predict(new_test, num_iteration=clf.best_iteration) / folds.n_splits



#print overall cross-validatino score

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:150].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,28))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('Features importance (averaged/folds)')

plt.tight_layout()

plt.savefig('FI.png')
df_submission = pd.DataFrame({'radiant_win_prob': predictions}, 

                                 index=df_test_features.index)

import datetime

submission_filename = 'submission_{}.csv'.format(

    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

df_submission.to_csv(submission_filename)

print('Submission saved to {}'.format(submission_filename))