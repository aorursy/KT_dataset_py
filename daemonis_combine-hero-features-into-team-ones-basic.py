import os

import pandas as pd

import numpy as np



PATH_TO_DATA = '../input/'



df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                             'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')
df_train_features.head()
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)
from sklearn.model_selection import ShuffleSplit, KFold

from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=17)
from sklearn.model_selection import train_test_split

import eli5

from IPython.display import display_html
def evaluate_model(train_df, test_df, target_df, model=model, cv=cv):



    X = train_df.values

    y = target_df['radiant_win'].values

    

    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    

    print (cv_scores, cv_scores.mean())

    

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)

    

    model.fit(X_train, y_train)

    

    display_html(eli5.show_weights(estimator=model, 

                  feature_names=train_df.columns.values, top=50))

    

    return model, cv_scores
model, cv_score_base = evaluate_model(df_train_features, df_test_features, df_train_targets)
def combine_numeric_features (df, feature_suffixes):

    for feat_suff in feature_suffixes:

        for team in 'r', 'd':

            players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...

            player_col_names = [f'{player}_{feat_suff}' for player in players] # e.g. r1_gold, r2_gold

            

            df[f'{team}_{feat_suff}_max'] = df[player_col_names].max(axis=1) # e.g. r_gold_max

            df[f'{team}_{feat_suff}_mean'] = df[player_col_names].mean(axis=1) # e.g. r_gold_mean

            df[f'{team}_{feat_suff}_min'] = df[player_col_names].min(axis=1) # e.g. r_gold_min

            

            df.drop(columns=player_col_names, inplace=True) # remove raw features from the dataset

    return df
numeric_features = ['kills', 'deaths', 'assists', 'denies', 'gold', 'xp', 'health', 'max_health', 'max_mana', 'level', 'towers_killed', 'stuns', 'creeps_stacked', 'camps_stacked', 'lh', 'rune_pickups', 'firstblood_claimed', 'teamfight_participation', 'roshans_killed', 'obs_placed', 'sen_placed']
df_train_features = combine_numeric_features(df_train_features, numeric_features)

df_test_features = combine_numeric_features(df_test_features, numeric_features)
df_train_features.head()
model, cv_score_num = evaluate_model(df_train_features, df_test_features, df_train_targets)
def make_coordinate_features(df):

    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...

        for player in players:

            df[f'{player}_distance'] = np.sqrt(df[f'{player}_x']**2 + df[f'{player}_y']**2)

            df.drop(columns=[f'{player}_x', f'{player}_y'], inplace=True)

    return df

df_train_features = make_coordinate_features(df_train_features)

df_test_features = make_coordinate_features(df_test_features)
df_train_features.head()
coord_features = ['distance']
df_train_features = combine_numeric_features(df_train_features, coord_features)

df_test_features = combine_numeric_features(df_test_features, coord_features)
df_train_features.head()
model, cv_score_coord = evaluate_model(df_train_features, df_test_features, df_train_targets)
from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import combinations
def hero_id_subset_analyzer(text):

    # it takes a string of hero ids (like '1 2 5 4 3') as input

    ids = set()

    for i in range(1, 4): # we need all subset of lenght 1-3. I think longer combinations are not relevant

        hero_ids = text.split(' ') # '1 2 5 4 3'-> ['1', '2', '5', '4', '3']

        hero_ids.sort() # sort them as '1 2 5 4 3' and '3 1 4 5 3' should produce the same set of tokens 

        combs = set(combinations(hero_ids, i)) # all combinations of length i e.g for 2 are: (1,2), (1,3)... (2,5)... etc

        ids = ids.union(combs)

    ids = { "_".join(item) for item in ids} # convert from lists to string e.g. (1,2) -> '1_2'

    return ids
# ngram range is (1,1) as all combinations are created by analyser

# 1000 features - I think it's enough to cover all heroes + popular combos

hero_id_vectorizer = TfidfVectorizer(ngram_range = (1, 1), max_features = 1000, tokenizer = lambda s: s.split(), analyzer=hero_id_subset_analyzer)
def replace_hero_ids (df, train=True, vectorizer=hero_id_vectorizer):



    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)] # r1, r2,...

        hero_columns = [f'{player}_hero_id' for player in players] # r1_hero_id,....

        

        # combine all hero id columns into one 

        df_hero_id_as_text = df[hero_columns].apply(lambda row: ' '.join([str(i) for i in row]), axis=1).tolist()

        

        if train:

            new_cols = pd.DataFrame(vectorizer.fit_transform(df_hero_id_as_text).todense(), columns = vectorizer.get_feature_names())

        else:

            new_cols = pd.DataFrame(vectorizer.transform(df_hero_id_as_text).todense(), columns = vectorizer.get_feature_names())

        

        # add index to vectorized dataset - needed for merge?

        new_cols['match_id_hash'] = df.index.values

        new_cols = new_cols.set_index('match_id_hash').add_prefix(f'{team}_hero_') # e.g.r_hero_10_21

        

        df = pd.merge(df, new_cols, on='match_id_hash')

        df.drop(columns=hero_columns, inplace=True)

    return df
df_train_features = replace_hero_ids(df_train_features)

df_test_features = replace_hero_ids(df_test_features, train=False)
df_train_features.head()
model, cv_score_hero = evaluate_model(df_train_features, df_test_features, df_train_targets)
X = df_train_features.values

y = df_train_targets['radiant_win'].values
%time

model.fit(X, y)
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