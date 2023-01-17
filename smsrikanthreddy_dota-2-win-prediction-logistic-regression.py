#libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.model_selection import ShuffleSplit, KFold

from sklearn.model_selection import cross_val_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time

PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction'

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA,'train_features.csv'), index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')
df_train_features.shape, df_train_targets.shape
df_train_features.head()
df_train_targets.head()
X = df_train_features.values

y = df_train_targets['radiant_win'].values
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3,random_state=17)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape
np.bincount(y)
#logistic Regression

C = 1

penalty = 'l2'

max_iter = 100

solver = 'liblinear'

random_state = 17

n_jobs = -1

verbose = 1



clf_lr = LogisticRegression(C=C,

                            penalty=penalty,

                            max_iter=max_iter, 

                            random_state=random_state,

                            verbose=verbose,

                            n_jobs=n_jobs,

                           solver=solver)
%%time

clf_lr.fit(x_train, y_train)

y_pred = clf_lr.predict(x_valid)

print('Log Regression validation roc_auc score {} '.format(roc_auc_score(y_pred, y_valid)))
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=17)
%%time

#calcuate ROC-AUC for each split

#logistic Regression

C = 1

penalty = 'l2'

max_iter = 50

solver = 'liblinear'

random_state = 17

n_jobs = -1

verbose = 1



clf_lr_1 = LogisticRegression(C=C,

                            penalty=penalty,

                            max_iter=max_iter, 

                            random_state=random_state,

                            verbose=verbose,

                            n_jobs=n_jobs,

                           solver=solver)



cv_scores_lr1 = cross_val_score(clf_lr_1, X, y, cv=cv, scoring='roc_auc')
%%time

#logistic Regression



C = 0.1

penalty = 'l2'

solver = 'saga'

max_iter = 150

random_state = 17

n_jobs = -1

verbose = 1

class_weight = 'balanced'



clf_lr_2 = LogisticRegression(C=C,

                            penalty=penalty,

                            max_iter=max_iter, 

                            random_state=random_state,

                            verbose=verbose,

                            n_jobs=n_jobs,

                            class_weight=class_weight,

                            solver=solver)



# calcuate ROC-AUC for each split

cv_scores_lr2 = cross_val_score(clf_lr_2, X, y, cv=cv, scoring='roc_auc')
cv_scores_lr2 > cv_scores_lr1
import json

with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as f:

    # read the 18-th line

    for i in range(18):

        line = f.readline()

    

    # read JSON into a Python object 

    match = json.loads(line)
player = match['players'][2]

#player
player['kills'], player['deaths'], player['assists']
player['ability_uses']
%matplotlib inline

from matplotlib import pyplot as plt
for player in match['players']:

    plt.plot(player['times'], player['gold_t'])

    

plt.title('Gold change for all players');
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
df_train_features_extended.head()
df_train_features_extended.shape
%%time

cv_scores_extended = cross_val_score(clf_lr_2, df_train_features_extended, y, cv=cv, scoring='roc_auc')
print('Base features: mean={} scores={}'.format(cv_scores_lr2.mean(), 

                                                cv_scores_lr2))

print('Extended features: mean={} scores={}'.format(cv_scores_extended.mean(), 

                                                    cv_scores_extended))
cv_scores_extended > cv_scores_lr2
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'),index_col='match_id_hash')
%%time

# Build the same features for the test set

df_test_features_extended = df_test_features.copy()

add_new_features(df_test_features_extended, os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))
clf_lr_2.fit(df_train_features_extended.values, y)

df_submission_base = pd.DataFrame(

    {'radiant_win_prob': clf_lr_2.predict_proba(df_test_features_extended.values)[:, 1]}, 

    index=df_test_features.index,

)

df_submission_base.to_csv('submission.csv')