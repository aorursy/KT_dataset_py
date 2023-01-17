import pandas as pd

import numpy as np

from catboost import CatBoostClassifier, Pool



from sklearn.model_selection import train_test_split
SEED = 1981

index_col = 'match_id_hash'

PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction/'
df_train = pd.read_csv(PATH_TO_DATA + 'train_features.csv', index_col=index_col)

df_targets = pd.read_csv(PATH_TO_DATA + 'train_targets.csv', index_col=index_col)
X_train, X_valid, y_train, y_valid = train_test_split(df_train, df_targets['radiant_win'].astype('int'), test_size=0.2, random_state=SEED)
cat_features = []

params_cb = {

    'loss_function': 'Logloss',

    'eval_metric': 'AUC',

    'verbose': 100,

    'random_seed': SEED,

    'thread_count': -1,

    'iterations':1000,

}



model_cb = CatBoostClassifier(**params_cb)
model_cb.fit(

    X_train,

    y_train,

    cat_features=cat_features,

    eval_set=(X_valid, y_valid),

    logging_level='Verbose',

    plot=True)
df_targets_upsample = df_targets.reset_index()

df_targets_upsample['match_id_hash'] = df_targets_upsample['match_id_hash'] + '_up'

df_targets_upsample.set_index('match_id_hash', inplace=True)

df_targets_upsample['radiant_win'] = df_targets_upsample['radiant_win'].map({False: True, True: False})



df_targets = df_targets.append(df_targets_upsample, sort=False)
df_train_upsample = df_train.reset_index()

df_train_upsample['match_id_hash'] = df_train_upsample['match_id_hash'] + '_up'

df_train_upsample.set_index('match_id_hash', inplace=True)
columns_names = df_train_upsample.columns

col_dict = {}

for col in columns_names:

    new_col = col

    if col[0] == 'd':

        new_col = col.replace('d', 'r', 1)



    elif col[0] == 'r':

        new_col = col.replace('r', 'd', 1)

    col_dict[col] = new_col
for i in range(1, 6):

    df_train_upsample[f'd{i}_y'] = 186 - df_train_upsample[f'd{i}_y'] + 70

    df_train_upsample[f'r{i}_y'] = 186 - df_train_upsample[f'r{i}_y'] + 70

    

    df_train_upsample[f'd{i}_x'] = 186 - df_train_upsample[f'd{i}_x'] + 66

    df_train_upsample[f'r{i}_x'] = 186 - df_train_upsample[f'r{i}_x'] + 66    
df_train_upsample.rename(columns=col_dict, inplace=True)
df_train = df_train.append(df_train_upsample, sort=False)
X_train, X_valid, y_train, y_valid = train_test_split(df_train, df_targets['radiant_win'].astype('int'), test_size=0.2, random_state=SEED)
model_cb.fit(

    X_train,

    y_train,

    cat_features=cat_features,

    eval_set=(X_valid, y_valid),

    logging_level='Verbose',

    plot=True)