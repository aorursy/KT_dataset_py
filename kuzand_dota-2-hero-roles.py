import pandas as pd

import numpy as np

import time

import datetime

import pytz

import os



# Sklearn stuff

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer
PATH_TO_DATA = '../input/'

SEED = 17
def write_to_submission_file(predicted_labels, filename='submission'):

    df_submission = pd.DataFrame({'radiant_win_prob': predicted_labels}, 

                                     index=df_test_features.index)



    submission_filename = '{}_{}.csv'.format(filename,

        datetime.datetime.now(tz=pytz.timezone('Europe/Athens')).strftime('%Y-%m-%d_%H-%M-%S'))

    

    df_submission.to_csv(submission_filename)

    

    print('Submission saved to {}'.format(submission_filename))
# Train dataset

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 

                                    index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), 

                                   index_col='match_id_hash')



y_train = df_train_targets['radiant_win'].map({True: 1, False: 0})



# Test dataset

df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 

                                   index_col='match_id_hash')
df_full_features = pd.concat([df_train_features, df_test_features])



# Index to split the training and test data sets

idx_split = df_train_features.shape[0]



heroes_df = df_full_features[[f'{t}{i}_hero_id' for t in ['r', 'd'] for i in range(1, 6)]]
hero_id_2_roles = {

                 1: ['Carry', 'Escape', 'Nuker'],

                 2: ['Initiator', 'Durable', 'Disabler', 'Jungler'],

                 3: ['Support', 'Disabler', 'Nuker', 'Durable'],

                 4: ['Carry', 'Disabler', 'Jungler', 'Nuker', 'Initiator'],

                 5: ['Support', 'Disabler', 'Nuker', 'Jungler'],

                 6: ['Carry', 'Disabler', 'Pusher'],

                 7: ['Support', 'Initiator', 'Disabler', 'Nuker'],

                 8: ['Carry', 'Pusher', 'Escape'],

                 9: ['Carry', 'Support', 'Escape', 'Nuker', 'Disabler'],

                 10: ['Carry', 'Escape', 'Durable', 'Nuker', 'Disabler'],

                 11: ['Carry', 'Nuker'],

                 12: ['Carry', 'Escape', 'Pusher', 'Nuker'],

                 13: ['Initiator', 'Disabler', 'Escape', 'Nuker'],

                 14: ['Disabler', 'Initiator', 'Durable', 'Nuker'],

                 15: ['Carry', 'Durable', 'Nuker', 'Pusher'],

                 16: ['Initiator', 'Disabler', 'Nuker', 'Escape', 'Jungler'],

                 17: ['Carry', 'Escape', 'Nuker', 'Initiator', 'Disabler'],

                 18: ['Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker'],

                 19: ['Carry', 'Nuker', 'Pusher', 'Initiator', 'Durable', 'Disabler'],

                 20: ['Support', 'Initiator', 'Disabler', 'Nuker', 'Escape'],

                 21: ['Carry', 'Support', 'Disabler', 'Escape', 'Nuker'],

                 22: ['Nuker'],

                 23: ['Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker'],

                 25: ['Support', 'Carry', 'Nuker', 'Disabler'],

                 26: ['Support', 'Disabler', 'Nuker', 'Initiator'],

                 27: ['Support', 'Pusher', 'Disabler', 'Nuker', 'Initiator'],

                 28: ['Carry', 'Durable', 'Initiator', 'Disabler', 'Escape'],

                 29: ['Initiator', 'Durable', 'Disabler', 'Nuker'],

                 30: ['Support', 'Nuker', 'Disabler'],

                 31: ['Support', 'Nuker'],

                 32: ['Carry', 'Escape', 'Disabler'],

                 33: ['Disabler', 'Jungler', 'Initiator', 'Pusher'],

                 34: ['Carry', 'Nuker', 'Pusher'],

                 35: ['Carry', 'Nuker'],

                 36: ['Carry', 'Nuker', 'Durable', 'Disabler'],

                 37: ['Support', 'Initiator', 'Disabler'],

                 38: ['Initiator', 'Disabler', 'Durable', 'Nuker'],

                 39: ['Carry', 'Nuker', 'Escape'],

                 40: ['Support', 'Nuker', 'Initiator', 'Pusher', 'Disabler'],

                 41: ['Carry', 'Initiator', 'Disabler', 'Escape', 'Durable'],

                 42: ['Carry', 'Support', 'Durable', 'Disabler', 'Initiator'],

                 43: ['Carry', 'Pusher', 'Nuker', 'Disabler'],

                 44: ['Carry', 'Escape'],

                 45: ['Nuker', 'Pusher'],

                 46: ['Carry', 'Escape'],

                 47: ['Carry', 'Durable', 'Initiator', 'Disabler'],

                 48: ['Carry', 'Nuker', 'Pusher'],

                 49: ['Carry', 'Pusher', 'Durable', 'Disabler', 'Initiator', 'Nuker'],

                 50: ['Support', 'Nuker', 'Disabler'],

                 51: ['Initiator', 'Disabler', 'Durable', 'Nuker'],

                 52: ['Carry', 'Support', 'Nuker', 'Pusher', 'Disabler'],

                 53: ['Carry', 'Jungler', 'Pusher', 'Escape', 'Nuker'],

                 54: ['Carry', 'Durable', 'Jungler', 'Escape', 'Disabler'],

                 55: ['Initiator', 'Jungler', 'Escape', 'Disabler'],

                 56: ['Carry', 'Escape', 'Pusher'],

                 57: ['Support', 'Durable', 'Nuker'],

                 58: ['Support', 'Jungler', 'Pusher', 'Durable', 'Disabler'],

                 59: ['Carry', 'Durable', 'Initiator'],

                 60: ['Carry', 'Initiator', 'Durable', 'Disabler', 'Nuker'],

                 61: ['Carry', 'Pusher', 'Escape', 'Nuker'],

                 62: ['Escape', 'Nuker'],

                 63: ['Carry', 'Escape'],

                 64: ['Support', 'Nuker', 'Pusher', 'Disabler'],

                 65: ['Initiator', 'Jungler', 'Disabler', 'Escape'],

                 66: ['Support', 'Jungler', 'Pusher'],

                 67: ['Carry', 'Durable', 'Escape'],

                 68: ['Support', 'Disabler', 'Nuker'],

                 69: ['Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker'],

                 70: ['Carry', 'Jungler', 'Durable', 'Disabler'],

                 71: ['Carry', 'Initiator', 'Disabler', 'Durable', 'Escape'],

                 72: ['Carry', 'Nuker', 'Disabler'],

                 73: ['Carry', 'Support', 'Durable', 'Disabler', 'Initiator', 'Nuker'],

                 74: ['Carry', 'Nuker', 'Disabler', 'Escape', 'Pusher'],

                 75: ['Carry', 'Support', 'Disabler', 'Initiator', 'Nuker'],

                 76: ['Carry', 'Nuker', 'Disabler'],

                 77: ['Carry', 'Pusher', 'Jungler', 'Durable', 'Escape'],

                 78: ['Carry', 'Initiator', 'Durable', 'Disabler', 'Nuker'],

                 79: ['Support', 'Disabler', 'Initiator', 'Nuker'],

                 80: ['Carry', 'Pusher', 'Jungler', 'Durable'],

                 81: ['Carry', 'Disabler', 'Durable', 'Pusher', 'Initiator'],

                 82: ['Carry', 'Escape', 'Nuker', 'Disabler', 'Initiator', 'Pusher'],

                 83: ['Support', 'Initiator', 'Durable', 'Disabler', 'Escape'],

                 84: ['Support', 'Nuker', 'Disabler', 'Durable', 'Initiator'],

                 85: ['Support', 'Durable', 'Disabler', 'Nuker'],

                 86: ['Support', 'Disabler', 'Nuker'],

                 87: ['Support', 'Disabler', 'Nuker', 'Initiator'],

                 88: ['Disabler', 'Nuker', 'Initiator', 'Escape'],

                 89: ['Carry', 'Support', 'Pusher', 'Disabler', 'Initiator', 'Escape'],

                 90: ['Support', 'Nuker', 'Disabler', 'Jungler'],

                 91: ['Support', 'Escape', 'Nuker'],

                 92: ['Support', 'Nuker', 'Durable', 'Disabler', 'Pusher'],

                 93: ['Carry', 'Escape', 'Disabler', 'Nuker'],

                 94: ['Carry', 'Disabler', 'Durable'],

                 95: ['Carry', 'Pusher', 'Disabler', 'Durable'],

                 96: ['Durable', 'Initiator', 'Disabler', 'Nuker', 'Escape'],

                 97: ['Initiator', 'Disabler', 'Nuker', 'Escape'],

                 98: ['Nuker', 'Durable', 'Escape'],

                 99: ['Carry', 'Durable', 'Initiator', 'Nuker'],

                 100: ['Initiator', 'Disabler', 'Nuker'],

                 101: ['Support', 'Nuker', 'Disabler'],

                 102: ['Support', 'Carry', 'Durable'],

                 103: ['Initiator', 'Disabler', 'Nuker', 'Durable'],

                 104: ['Carry', 'Disabler', 'Initiator', 'Durable', 'Nuker'],

                 105: ['Nuker', 'Disabler'],

                 106: ['Carry', 'Escape', 'Nuker', 'Disabler', 'Initiator'],

                 107: ['Nuker', 'Escape', 'Disabler', 'Initiator', 'Durable'],

                 108: ['Support', 'Nuker', 'Disabler', 'Durable', 'Escape'],

                 109: ['Carry', 'Pusher', 'Nuker'],

                 110: ['Support', 'Nuker', 'Initiator', 'Escape', 'Disabler'],

                 111: ['Support', 'Nuker', 'Disabler', 'Escape'],

                 112: ['Support', 'Disabler', 'Nuker'],

                 113: ['Carry', 'Escape', 'Nuker'],

                 114: ['Carry', 'Escape', 'Disabler', 'Initiator'],

                 119: ['Support', 'Nuker', 'Disabler', 'Escape'],

                 120: ['Carry', 'Nuker', 'Disabler', 'Durable', 'Escape', 'Initiator']}
set(role for roles in hero_id_2_roles.values() for role in roles)
def bag_of_hero_roles(df):   

    r_df = df[[f'r{i}_hero_id' for i in range(1, 6)]]

    d_df = df[[f'd{i}_hero_id' for i in range(1, 6)]]

    

    r_roles = r_df.applymap(lambda x: ' '.join(hero_id_2_roles[x])).apply(lambda x: ' '.join(x), axis=1).values

    d_roles = d_df.applymap(lambda x: ' '.join(hero_id_2_roles[x])).apply(lambda x: ' '.join(x), axis=1).values

    

    vectorizer_r = CountVectorizer()

    vectorizer_d = CountVectorizer()

    

    X_r = vectorizer_r.fit_transform(r_roles)

    X_d = vectorizer_d.fit_transform(d_roles)

    

    return pd.concat([pd.DataFrame(X_r.toarray(),

                                   columns=[f'r_{role[0]}' for role in sorted(vectorizer_r.vocabulary_.items())],

                                   index=df.index),

                      pd.DataFrame(X_d.toarray(),

                                   columns=[f'd_{role[0]}' for role in sorted(vectorizer_d.vocabulary_.items())],

                                   index=df.index)], axis=1)

    
bohr = bag_of_hero_roles(heroes_df)
bohr.head()
def logit_cv(X_train, y_train, cv, random_state=SEED):

    logit = LogisticRegression(random_state=SEED, solver='liblinear')



    c_values = np.logspace(-2, 1, 20)



    logit_grid_searcher = GridSearchCV(estimator=logit, param_grid={'C': c_values},

                                       scoring='roc_auc',return_train_score=False, cv=cv,

                                       n_jobs=-1, verbose=0)



    logit_grid_searcher.fit(X_train, y_train)

    

    cv_scores = []

    for i in range(logit_grid_searcher.n_splits_):

        cv_scores.append(logit_grid_searcher.cv_results_[f'split{i}_test_score'][logit_grid_searcher.best_index_])

    print(f'CV scores: {cv_scores}')

    print(f'Mean: {np.mean(cv_scores)}, std: {np.std(cv_scores)}\n')

    

    return logit_grid_searcher.best_estimator_, np.array(cv_scores) 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
logit = logit_cv(bohr[:idx_split], y_train, cv=skf, random_state=SEED)[0]
logit.fit(bohr[:idx_split], y_train)

logit_pred = logit.predict_proba(bohr[idx_split:])[:, 1]



write_to_submission_file(logit_pred, filename='logit_hero_roles')
r_roles = [role for role in bohr.columns if role[0] == 'r']

d_roles = [role for role in bohr.columns if role[0] == 'd']



bohr_diff = pd.DataFrame(bohr[r_roles].values - bohr[d_roles].values,

                         columns=[role[2:] for role in bohr.columns if role[0] == 'r'],

                         index=bohr.index)
bohr_diff.head()
logit = logit_cv(bohr_diff[:idx_split], y_train, cv=skf, random_state=SEED)[0]